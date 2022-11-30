"""geometry module"""
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
from mindspore import ops


def index(feat, uv_p):
    """
    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [0, 1]
    :return: [B, C, N] image features at the uv coordinates
    """
    uv_p = ops.transpose(uv_p, (0, 2, 1))

    (batch, n_p, _) = uv_p.shape
    channel = feat.shape[1]

    if uv_p.shape[-1] == 3:
        # uv = uv[:,:,[2,1,0]]
        # uv = uv * torch.tensor([1.0,-1.0,1.0]).type_as(uv)[None,None,...]
        uv_p = ops.expand_dims(ops.expand_dims(uv_p, 2), 3)  # [B, N, 1, 1, 3]
    else:
        uv_p = ops.expand_dims(uv_p, 2)  # [B, N, 1, 2]

    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = ops.grid_sample(feat, uv_p, align_corners=True)  # [B, C, N, 1]
    return samples.view(batch, channel, n_p)  # [B, C, N]


def orthogonal(points, calibrations, transforms=None):
    """
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 3, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    """
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    bmm = ops.BatchMatMul()
    pts = trans + bmm(rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = shift + bmm(scale, pts[:, :2, :])
    return pts


def perspective(points, calibrations, transforms=None):
    """
    Compute the perspective projections of 3D points into the image plane by given projection matrix
    :param points: [Bx3xN] Tensor of 3D points
    :param calibrations: [Bx3x4] Tensor of projection matrix
    :param transforms: [Bx2x3] Tensor of image transform matrix
    :return: xy: [Bx2xN] Tensor of xy coordinates in the image plane
    """
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    bmm = ops.BatchMatMul()
    homo = trans + bmm(rot, points)  # [B, 3, N]
    x_y = homo[:, :2, :] / homo[:, 2:3, :]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        x_y = shift + bmm(scale, x_y)

    x_y_z = ops.concat([x_y, homo[:, 2:3, :]], axis=1)
    return x_y_z
