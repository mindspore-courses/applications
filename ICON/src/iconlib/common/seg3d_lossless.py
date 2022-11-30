"""sdf"""
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import mindspore as ms
from mindspore import Tensor, nn, Parameter
import torch
import numpy as np
import mcubes
from kaolin.ops.conversions import voxelgrids_to_trianglemeshes

from .seg3d_utils import (
    create_grid_3d,
    SmoothConv3D,
)


class Seg3dLossless(nn.Cell):
    """Implicit reconstruct module"""

    def __init__(
            self,
            query_func,
            b_min,
            b_max,
            resolutions,
            channels=1,
            balance_value=0.5,
            align_corners=False,
            visualize=False,
            debug=False,
            use_cuda_impl=False,
            faster=False,
            use_shadow=False,
    ):

        """
        align_corners: same with how you process gt. (grid_sample / interpolate)
        """

        super().__init__()

        self.query_func = query_func
        self.b_min = Parameter(
            ms.ops.expand_dims(Tensor(b_min), axis=1), requires_grad=False
        )
        self.b_max = Parameter(
            ms.ops.expand_dims(Tensor(b_max), axis=1), requires_grad=False
        )

        if isinstance(resolutions[0]) is int:
            resolutions = Tensor([(res, res, res) for res in resolutions])
        else:
            resolutions = Tensor(resolutions)

        self.resolutions = Parameter(resolutions, requires_grad=False)
        self.batchsize = self.b_min.size(0)
        assert self.batchsize == 1
        self.balance_value = balance_value
        self.channels = channels
        assert self.channels == 1
        self.align_corners = align_corners
        self.visualize = visualize
        self.debug = debug
        self.use_cuda_impl = use_cuda_impl
        self.faster = faster
        self.use_shadow = use_shadow

        for resolution in resolutions:
            assert (
                resolution[0] % 2 == 1 and resolution[1] % 2 == 1
            ), f"resolution {resolution} need to be odd because of align_corner."

        # init first resolution
        init_coords = create_grid_3d(
            0, resolutions[-1] - 1, steps=resolutions[0]
        )  # [N, 3]
        init_coords = ms.ops.tile(
            ms.ops.expand_dims(init_coords, axis=0), (self.batchsize, 1, 1)
        )  # [bz, N, 3]
        self.init_coords = Parameter(init_coords, requires_grad=False)

        # some useful tensors
        calculated = ms.ops.zeros(
            (self.resolutions[-1][2], self.resolutions[-1][1], self.resolutions[-1][0]),
            dtype=ms.bool_,
        )
        self.calculated = Parameter(calculated, requires_grad=False)

        gird8_offsets = (
            ms.ops.stack(
                ms.ops.meshgrid(
                    [Tensor([-1, 0, 1]), Tensor([-1, 0, 1]), Tensor([-1, 0, 1])]
                )
            )
            .astype(ms.int32)
            .view(3, -1)
            .t()
        )  # [27, 3]
        self.gird8_offsets = Parameter(gird8_offsets, requires_grad=False)

        # smooth convs
        self.smooth_conv3x3 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=3)
        self.smooth_conv5x5 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=5)
        self.smooth_conv7x7 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=7)
        self.smooth_conv9x9 = SmoothConv3D(in_channels=1, out_channels=1, kernel_size=9)

    def batch_eval(self, coords, **kwargs):
        """
        coords: in the coordinates of last resolution
        **kwargs: for query_func
        """
        coords = ms.ops.stop_gradient(coords)
        # normalize coords to fit in [b_min, b_max]
        if self.align_corners:
            coords_2d = coords.astype(ms.float32) / (self.resolutions[-1] - 1)
        else:
            step = 1.0 / self.resolutions[-1].astype(ms.float32)
            coords_2d = coords.astype(ms.float32) / self.resolutions[-1] + step / 2
        coords_2d = coords_2d * (self.b_max - self.b_min) + self.b_min
        # query function
        occupancys = self.query_func(**kwargs, points=coords_2d)
        if isinstance(occupancys) is list:
            occupancys = ms.ops.stack(occupancys)  # [bz, C, N]
        assert (
            len(occupancys.shape) == 3
        ), "query_func should return a occupancy with shape of [bz, C, N]"
        return occupancys

    def export_mesh(self, occupancys):
        """Export body mesh"""
        occupancys = torch.Tensor(occupancys.asnumpy())

        final = occupancys[:-1, :-1, :-1].contiguous()

        if final.shape[0] > 256:
            # for voxelgrid larger than 256^3, the required GPU memory will be > 9GB
            # thus we use CPU marching_cube to avoid "CUDA out of memory"
            occu_arr = final.detach().cpu().numpy()  # non-smooth surface
            # occu_arr = mcubes.smooth(final.detach().cpu().numpy())  # smooth surface
            vertices, triangles = mcubes.marching_cubes(occu_arr, self.balance_value)
            verts = torch.as_tensor(vertices[:, [2, 1, 0]])
            faces = torch.as_tensor(triangles.astype(np.long), dtype=torch.long)[
                :, [0, 2, 1]
            ]
        else:
            torch.cuda.empty_cache()
            vertices, triangles = voxelgrids_to_trianglemeshes(final.unsqueeze(0))
            verts = vertices[0][:, [2, 1, 0]].cpu()
            faces = triangles[0][:, [0, 2, 1]].cpu()

        verts = Tensor(verts.numpy())
        faces = Tensor(faces.numpy())

        return verts, faces

    def construct(self, **kwargs):
        """output occupancy field would be: (bz, C, res, res)"""

        calculated = self.calculated.copy()

        for resolution in self.resolutions:
            width, height, direction = resolution
            stride = (self.resolutions[-1] - 1) / (resolution - 1)

            if self.visualize:
                this_stage_coords = []

            # first step
            if ms.ops.equal(resolution, self.resolutions[0]):
                coords = self.init_coords.copy()  # torch.long
                occupancys = self.batch_eval(coords, **kwargs)
                occupancys = occupancys.view(
                    self.batchsize, self.channels, direction, height, width
                )

                coords_accum = coords / stride
                calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True

            # next steps
            else:
                coords_accum *= 2

                # here true is correct!
                valid = ms.ops.interpolate(
                    (occupancys > self.balance_value).astype(ms.float32),
                    size=(direction, height, width),
                    mode="bilinear",
                )

                # here true is correct!
                occupancys = ms.ops.interpolate(
                    occupancys.astype(ms.float32),
                    size=(direction, height, width),
                    mode="bilinear",
                )

                is_boundary = (valid > 0.0) & (valid < 1.0)

                # TODO
                if self.use_shadow and ms.ops.equal(resolution, self.resolutions[-1]):
                    # larger z means smaller depth here
                    depth_res = resolution[2].item()
                    depth_index = ms.ops.linspace(
                        0, depth_res - 1, num=depth_res
                    ).astype(occupancys.device.dtype)
                    argmax = ms.ops.ArgMaxWithValue(axis=-1, keep_dims=True)
                    depth_index_max = (
                        argmax((occupancys > self.balance_value) * (depth_index + 1),)[
                            0
                        ]
                        - 1
                    )
                    shadow = depth_index < depth_index_max
                    is_boundary[shadow] = False
                    is_boundary = is_boundary[0, 0]
                else:
                    is_boundary = (
                        self.smooth_conv3x3(is_boundary.astype(ms.float32)) > 0
                    )[0, 0]
                    # is_boundary = is_boundary[0, 0]

                is_boundary[
                    coords_accum[0, :, 2], coords_accum[0, :, 1], coords_accum[0, :, 0]
                ] = False
                point_coords = ms.ops.expand_dims(
                    ms.ops.nonzero(ms.ops.transpose(is_boundary, (2, 1, 0))), axis=0
                )
                point_indices = (
                    point_coords[:, :, 2] * height * width
                    + point_coords[:, :, 1] * width
                    + point_coords[:, :, 0]
                )

                repeat, channel, direction, height, width = occupancys.shape
                # interpolated value
                occupancys_interp = ms.ops.gather_elements(
                    ms.ops.reshape(occupancys, (repeat, channel, direction * height * width)),
                    2,
                    ms.ops.expand_dims(point_indices, axis=1),
                )

                # inferred value
                coords = point_coords * stride

                if coords.shape[1] == 0:
                    continue
                occupancys_topk = self.batch_eval(coords, **kwargs)

                # put mask point predictions to the right places on the upsampled grid.
                scatter_nd_add = ms.ops.ScatterNdAdd()
                repeat, channel, direction, height, width = occupancys.shape
                point_indices = ms.ops.broadcast_to(
                    ms.ops.expand_dims(point_indices, axis=1), (-1, C, -1)
                )
                occupancys = scatter_nd_add(
                    (
                        ms.ops.reshape(
                            occupancys, (repeat, channel, direction * height * width)
                        ),
                        2,
                        point_indices,
                        occupancys_topk,
                    ).view(repeat, channel, direction, height, width)
                )

                # conflicts
                conflicts = (
                    (occupancys_interp - self.balance_value)
                    * (occupancys_topk - self.balance_value)
                    < 0
                )[0, 0]

                unique = ms.ops.Unique()

                voxels = coords / stride
                coords_accum = unique(ms.ops.concat([voxels, coords_accum], axis=1))
                calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True

                while ms.ops.reduce_sum()(conflicts) > 0:
                    if self.use_shadow and ms.ops.equal(
                            resolution, self.resolutions[-1]
                    ):
                        break

                    conflicts_coords = coords[0, conflicts, :]

                    conflicts_boundary = unique(
                        ms.ops.reshape(
                            (
                                conflicts_coords.astype(ms.int32)
                                + ms.ops.expand_dims(self.gird8_offsets, axis=1)
                                * stride.astype(ms.int32)
                            ),
                            (-1, 3),
                        ).astype(ms.int64)
                    )
                    conflicts_boundary[:, 0] = ms.ops.clip_by_value(
                        conflicts_boundary[:, 0], 0, calculated.shape[2] - 1
                    )
                    conflicts_boundary[:, 1] = ms.ops.clip_by_value(
                        conflicts_boundary[:, 1], 0, calculated.shape[1] - 1
                    )
                    conflicts_boundary[:, 2] = ms.ops.clip_by_value(
                        conflicts_boundary[:, 2], 0, calculated.shape[0] - 1
                    )

                    coords = conflicts_boundary[
                        calculated[
                            conflicts_boundary[:, 2],
                            conflicts_boundary[:, 1],
                            conflicts_boundary[:, 0],
                        ]
                    ]

                    coords = ms.ops.expand_dims(coords, axis=0)
                    point_coords = coords / stride
                    point_indices = (
                        point_coords[:, :, 2] * height * width
                        + point_coords[:, :, 1] * width
                        + point_coords[:, :, 0]
                    )

                    repeat, channel, direction, height, width = occupancys.shape
                    # interpolated value
                    occupancys_interp = ms.ops.gather_elements(
                        ms.ops.reshape(
                            occupancys, (repeat, channel, direction * height * width)
                        ),
                        2,
                        ms.ops.expand_dims(point_indices, axis=1),
                    )

                    # inferred value
                    coords = point_coords * stride

                    if coords.shape[1] == 0:
                        break
                    occupancys_topk = self.batch_eval(coords, **kwargs)
                    if self.visualize:
                        this_stage_coords.append(coords)

                    # conflicts
                    conflicts = (
                        (occupancys_interp - self.balance_value)
                        * (occupancys_topk - self.balance_value)
                        < 0
                    )[0, 0]

                    # put mask point predictions to the right places on the upsampled grid.
                    point_indices = ms.ops.broadcast_to(
                        ms.ops.expand_dims(point_indices, axis=1), (-1, C, -1)
                    )
                    occupancys = ms.ops.scatter_nd_add(
                        (
                            ms.ops.reshape(
                                occupancys,
                                (repeat, channel, direction * height * width),
                            ),
                            2,
                            point_indices,
                            occupancys_topk,
                        ).view(repeat, channel, direction, height, width)
                    )

                    voxels = coords / stride
                    coords_accum = unique(ms.ops.concat([voxels, coords_accum], axis=1))
                    calculated[coords[0, :, 2], coords[0, :, 1], coords[0, :, 0]] = True

        return occupancys[0, 0]
