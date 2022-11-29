"""maf extractor"""
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
import os
import mindspore.nn as nn
import mindspore as ms
from mindspore import Tensor, Parameter, COOTensor
import mindspore.ops as ops
import numpy as np
import scipy
from iconlib.common.config import cfg
from iconlib.pymaf.utils.geometry import projection

import torch


class MAFExtractor(nn.Cell):
    """ Mesh-aligned Feature Extractor
    As discussed in the paper, we extract mesh-aligned features based
     on 2D projection of the mesh vertices.
    The features extrated from spatial feature maps will go through a MLP for dimension reduction.
    """

    def __init__(self):
        super().__init__()

        self.filters = []
        self.num_views = 1
        filter_channels = cfg.MODEL.PyMAF.MLP_DIM
        self.last_op = nn.ReLU()

        for l_p in range(0, len(filter_channels) - 1):
            if l_p != 0:
                self.filters.append(
                    nn.Conv1d(
                        filter_channels[l_p] + filter_channels[0],
                        filter_channels[l_p + 1],
                        1,
                        has_bias=True,
                    )
                )
            else:
                self.filters.append(
                    nn.Conv1d(
                        filter_channels[l_p], filter_channels[l_p + 1], 1, has_bias=True
                    )
                )

            self.insert_child_to_cell(f"conv{l_p}", self.filters[l_p])

        self.im_feat = None
        self.cam = None

        # downsample SMPL mesh and assign part labels
        # from https://github.com/nkolot/GraphCMR/blob/master/data/mesh_downsampling.npz
        mesh_downsampling_path = os.path.join(
            os.path.dirname(__file__), "../../../data/pymaf_data/mesh_downsampling.npz"
        )
        smpl_mesh_graph = np.load(
            mesh_downsampling_path, allow_pickle=True, encoding="latin1"
        )

        d_p = smpl_mesh_graph["D"]  # shape: (2,)

        # downsampling
        pt_d = []
        for item in enumerate(d_p):
            i = item[0]
            d_p_p = scipy.sparse.coo_matrix(d_p[i])
            i = ms.ops.transpose(Tensor(np.array([d_p_p.row, d_p_p.col])), (1, 0))
            v_p = Tensor(d_p_p.data, dtype=ms.float32)
            pt_d.append(COOTensor(i, v_p, d_p_p.shape))

        # downsampling mapping from 6890 points to 431 points
        # ptD[0].to_dense() - Size: [1723, 6890]
        # ptD[1].to_dense() - Size: [431. 1723]
        d_map = ms.ops.matmul(pt_d[1].to_dense(), pt_d[0].to_dense())  # 6890 -> 431
        self.d_map = Parameter(d_map, requires_grad=False)

    def reduce_dim(self, feature):
        """
        Dimension reduction by multi-layer perceptrons
        :param feature: list of [B, C_s, N] point-wise features before dimension reduction
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        """
        y = feature
        tmpy = feature
        for item in enumerate(self.filters):
            i = item[0]
            layer = getattr(self, "conv" + str(i))
            y = layer(y if i == 0 else ops.concat([y, tmpy], axis=1))
            if i != len(self.filters) - 1:
                leaky_relu = nn.LeakyReLU()
                y = leaky_relu(y)
            if self.num_views > 1 and i == len(self.filters) // 2:
                reduce_mean = ops.ReduceMean()
                y = reduce_mean(y.view(-1, self.num_views, y.shape[1], y.shape[2]), 1)
                tmpy = reduce_mean(
                    feature.view(
                        -1, self.num_views, feature.shape[1], feature.shape[2]
                    ),
                    1,
                )

        y = self.last_op(y)

        y = y.view(y.shape[0], -1)
        return y

    def sampling(self, points, im_feat=None):
        """
        Given 2D points, sample the point-wise features for each point,
        the dimension of point-wise features will be reduced from C_s to C_p by MLP.
        Image features should be pre-computed before this call.
        :param points: [B, N, 2] image coordinates of points
        :im_feat: [B, C_s, H_s, W_s] spatial feature maps
        :return: [B, C_p x N] concatantion of point-wise features after dimension reduction
        """
        if im_feat is None:
            im_feat = self.im_feat

        points = torch.Tensor(points.asnumpy())
        im_feat = torch.Tensor(im_feat.asnumpy())

        point_feat = torch.nn.functional.grid_sample(
            im_feat, points.unsqueeze(2), align_corners=True
        )[..., 0]

        point_feat = Tensor(point_feat.cpu().numpy())

        mesh_align_feat = self.reduce_dim(point_feat)
        return mesh_align_feat

    def construct(self, p, s_feat=None, cam=None):
        """ Returns mesh-aligned features for the 3D mesh points.

        Args:
            p (tensor): [B, N_m, 3] mesh vertices
            s_feat (tensor): [B, C_s, H_s, W_s] spatial feature maps
            cam (tensor): [B, 3] camera
        Return:
            mesh_align_feat (tensor): [B, C_p x N_m] mesh-aligned features
        """
        if cam is None:
            cam = self.cam
        p_proj_2d = projection(p, cam, retain_z=False)
        mesh_align_feat = self.sampling(p_proj_2d, s_feat)
        return mesh_align_feat
