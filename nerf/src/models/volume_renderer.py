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
"""nerf volume renderer"""

import mindspore
import mindspore.ops.operations as P
from mindspore import nn

from nerf_utils import Embedder
from nerf_utils.sampler import sample_along_rays, sample_pdf

__all__ = ["VolumeRenderer"]


class VolumeRenderer(nn.Cell):
    """
    Volume Renderer architecture.

    Args:
        chunk (int): Number of rays processed in parallel, decrease if running out of memory.
        cap_n_samples (int): Number of coarse samples per ray for coarse net.
        cap_n_importance (int): Number of additional fine samples per ray for fine net.
        net_chunk (int): Number of pts sent through network in parallel, decrease if running out of memory.
        white_bkgd (bool): Set to render synthetic data on a white background (always use for DeepVoxels).
        model_coarse (nn.Cell): Coarse net.
        model_fine (nn.Cell, optional): Fine net, or None.
        embedder_p (Dict): Config for positional encoding for point.
        embedder_d (Dict): Config for positional encoding for view direction.
        near (float, optional): The near plane. Default: 0.0.
        far (float, optional): The far plane. Default: 1e6.

    Inputs:
        - **rays** (Tensor) - The ray tensor. (..., num_pts_per_ray, ray_batch_dims).

    Outputs:
        Tuple of 2 Tensor, the output tensors.

        - **fine_net_output** (Tensor, optional) - The fine net output features.
        - **coarse_net_output** (Tensor) - The coarse net output features.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> model = VolumeRenderer(1000, 6, 12, 1000, False, P.Identity(), P.Identity(), P.Identity(), P.Identity())
        >>> inputs = ms.numpy.randn(1, 1, 3)
        >>> outputs = model(inputs)
        ([[0.2384 0.8456 0.1273]], [[0.8653 0.1866 0.6382]])
    """

    def __init__(self,
                 chunk,
                 cap_n_samples,
                 cap_n_importance,
                 net_chunk,
                 white_bkgd,
                 model_coarse,
                 model_fine,
                 embedder_p,
                 embedder_d,
                 near=0.0,
                 far=1e6):
        super().__init__()

        self.chunk = chunk
        self.cap_n_samples = cap_n_samples
        self.cap_n_importance = cap_n_importance
        self.net_chunk = net_chunk
        self.white_bkgd = white_bkgd

        self.model_coarse = model_coarse
        self.model_fine = model_fine
        # embedder for positions
        self.embedder_p = Embedder(**embedder_p)
        # embedder for view-in directions
        self.embedder_d = Embedder(**embedder_d)

        self.near = near
        self.far = far

    def construct(self, rays):
        """Volume renderer construct."""
        return self.inference(rays)

    def inference(self, rays):
        """Volume renderer inference."""
        # make the number of rays be multiple of the chunk size
        cap_n_rays = (rays.shape[1] // self.chunk + 1) * self.chunk
        cap_n = self.cap_n_samples

        res_ls = {"rgb_map_coarse": [], "rgb_map_fine": []}

        for i in range(0, cap_n_rays, self.chunk):
            ray_origins, ray_dirs = rays[:, i:i + self.chunk, :]
            reshape_op = mindspore.ops.Reshape()
            view_dirs = reshape_op(
                ray_dirs / mindspore.numpy.norm(ray_dirs, axis=-1, keepdims=True),
                (-1, 3),
            )

            near, far = self.near * mindspore.numpy.ones_like(ray_dirs[..., :1]), self.far * mindspore.numpy.ones_like(
                ray_dirs[..., :1])
            cap_m = ray_origins.shape[0]
            if cap_m == 0:
                continue

            # stratified sampling along rays
            s_samples = sample_along_rays(near, far, cap_n)

            # position samples along rays
            unsqueeze_op = P.ExpandDims()
            pos_samples = unsqueeze_op(ray_origins,
                                       1) + unsqueeze_op(ray_dirs, 1) * unsqueeze_op(s_samples, 2)
            # expand ray directions to the same shape of samples
            expand_op = P.BroadcastTo(pos_samples.shape)
            dir_samples = expand_op(unsqueeze_op(view_dirs, 1))

            reshape_op = P.Reshape()
            pos_samples = reshape_op(pos_samples, (-1, 3))
            dir_samples = reshape_op(dir_samples, (-1, 3))

            # retrieve optic data from the network
            optic_d = self._run_network_model_coarse(pos_samples, dir_samples)
            optic_d = mindspore.numpy.reshape(optic_d, [cap_m, cap_n, 4])

            # composite optic data to generate a RGB image
            rgb_map_coarse, weights_coarse = self._composite(optic_d, s_samples, ray_dirs)

            if self.cap_n_importance > 0:
                z_vals_mid = 0.5 * (s_samples[..., 1:] + s_samples[..., :-1])
                z_samples = sample_pdf(z_vals_mid, weights_coarse[..., 1:-1], self.cap_n_importance)
                z_samples = mindspore.ops.stop_gradient(z_samples)

                sort_op = P.Sort(axis=-1)
                z_vals, _ = sort_op(P.Concat(-1)([s_samples, z_samples]))
                pts = (ray_origins[..., None, :] + ray_dirs[..., None, :] * z_vals[..., :, None]
                      )

                expand_op_2 = P.BroadcastTo(pts.shape)
                dir_samples = expand_op_2(unsqueeze_op(view_dirs, 1))

                pts = reshape_op(pts, (-1, 3))
                dir_samples = reshape_op(dir_samples, (-1, 3))

                optic_d = self._run_network_model_fine(pts, dir_samples)
                optic_d = reshape_op(optic_d, (cap_m, cap_n + self.cap_n_importance, 4))

                rgb_map_fine, _ = self._composite(optic_d, z_vals, ray_dirs)
            else:
                rgb_map_fine = rgb_map_coarse

            res_ls["rgb_map_coarse"].append(rgb_map_coarse)
            res_ls["rgb_map_fine"].append(rgb_map_fine)

        res = {}
        for k, v in res_ls.items():
            res[k] = P.Concat(0)(v)

        return res["rgb_map_fine"], res["rgb_map_coarse"]

    def _run_network_model_fine(self, pts, view_dirs):
        """Run fine model."""
        inputs_flat = pts
        embedded = self.embedder_p(inputs_flat)

        if view_dirs is not None:
            input_dirs_flat = view_dirs
            embedded_dirs = self.embedder_d(input_dirs_flat)
            embedded = P.Concat(-1)([embedded, embedded_dirs])

        chunk = self.net_chunk
        outputs_flat_ls = []
        for i in range(0, embedded.shape[0], chunk):
            outputs_flat_ls.append(self.model_fine(embedded[i:i + chunk]))
        outputs_flat = P.Concat(0)(outputs_flat_ls)
        return outputs_flat

    def _run_network_model_coarse(self, pts, view_dirs):
        """Run coarse model."""
        inputs_flat = pts
        embedded = self.embedder_p(inputs_flat)

        if view_dirs is not None:
            input_dirs_flat = view_dirs
            embedded_dirs = self.embedder_d(input_dirs_flat)
            embedded = P.Concat(-1)([embedded, embedded_dirs])

        chunk = self.net_chunk
        outputs_flat_ls = []
        for i in range(0, embedded.shape[0], chunk):
            outputs_flat_ls.append(self.model_coarse(embedded[i:i + chunk]))
        outputs_flat = P.Concat(0)(outputs_flat_ls)
        return outputs_flat

    def _transfer(self, optic_d, dists):
        """Transfer occupancy to alpha values."""
        sigmoid = P.Sigmoid()
        rgbs = sigmoid(optic_d[..., :3])
        alphas = 1.0 - P.Exp()(-1.0 * (P.ReLU()(optic_d[(..., 3)])) * dists)

        return rgbs, alphas

    def _composite(self, optic_d, s_samples, rays_d):
        """Composite the colors and densities."""
        # distances between each samples
        dists = s_samples[..., 1:] - s_samples[..., :-1]
        dists_list = (
            dists,
            (mindspore.numpy.ones([]) * 1e10).expand_as(dists[..., :1]),
        )
        dists = P.Concat(-1)(dists_list)

        dists = dists * mindspore.numpy.norm(rays_d[..., None, :], axis=-1)

        # retrieve display colors and alphas for each samples by a transfer function
        rgbs, alphas = self._transfer(optic_d, dists)

        weights = alphas * mindspore.numpy.cumprod(
            P.Concat(-1)([mindspore.numpy.ones((alphas.shape[0], 1)), 1.0 - alphas + 1e-10])[:, :-1],
            axis=-1,
        )
        sum_op = mindspore.ops.ReduceSum()
        rgb_map = sum_op(weights[..., None] * rgbs, -2)
        acc_map = sum_op(weights, -1)

        if self.white_bkgd:
            rgb_map = rgb_map + (1.0 - acc_map[..., None])

        return rgb_map, weights
