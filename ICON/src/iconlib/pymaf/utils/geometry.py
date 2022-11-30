"""geomerty module"""
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
import mindspore.ops as ops
import mindspore as ms
import mindspore.numpy as numpy
from mindspore import Tensor
from mindspore.ops.function import broadcast_to

expand_dims = ops.ExpandDims()


def rot6d_to_rotmat(x_p):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation
    Representations in Neural Networks", CVPR 2019
    Input:
        (B,6) Batch of 6-D rotation representations
    Output:
        (B,3,3) Batch of corresponding rotation matrices
    """
    x_p = x_p.view(-1, 3, 2)
    a_1 = x_p[:, :, 0]
    a_2 = x_p[:, :, 1]
    l2_normalize = ops.L2Normalize(axis=1)
    einsum = ops.Einsum("bi,bi->b")
    b_1 = l2_normalize(a_1)
    b_2 = l2_normalize(a_2 - expand_dims(einsum((b_1, a_2)), -1) * b_1)
    b_3 = numpy.cross(b_1, b_2)
    stack = ops.Stack(axis=-1)
    return stack([b_1, b_2, b_3])


def perspective_projection(
        points, rotation, translation, focal_length, camera_center, retain_z=False
):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    zeros = ops.Zeros()
    k_p = zeros((batch_size, 3, 3), ms.float32)
    k_p[:, 0, 0] = focal_length
    k_p[:, 1, 1] = focal_length
    k_p[:, 2, 2] = 1.0
    k_p[:, :-1, -1] = camera_center

    # Transform points
    einsum = ops.Einsum("bij,bkj->bki")
    points = einsum((rotation, points))
    points = points + expand_dims(translation, 1)

    # Apply perspective distortion
    projected_points = points / expand_dims(points[:, :, -1], -1)

    # Apply camera intrinsics
    projected_points = einsum((k_p, projected_points))

    if retain_z:
        return projected_points
    return projected_points[:, :, :-1]


def projection(pred_joints, pred_camera, retain_z=False):
    """projection"""
    pred_cam_t = ops.stack(
        (
            pred_camera[:, 1],
            pred_camera[:, 2],
            2 * 5000.0 / (224.0 * pred_camera[:, 0] + 1e-9),
        ),
        axis=-1,
    )
    batch_size = pred_joints.shape[0]
    zeros = ops.Zeros()
    camera_center = zeros((batch_size, 2), ms.float32)
    eye = ops.Eye()
    pred_keypoints_2d = perspective_projection(
        pred_joints,
        rotation=broadcast_to(
            expand_dims(eye(3, 3, ms.float32), 0), (batch_size, -1, -1)
        ),
        translation=pred_cam_t,
        focal_length=5000.0,
        camera_center=camera_center,
        retain_z=retain_z,
    )
    # Normalize keypoints to [-1,1]
    pred_keypoints_2d = pred_keypoints_2d / (224.0 / 2.0)
    return pred_keypoints_2d


def rotation_matrix_to_quaternion(rotation_matrix, eps=1e-6):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to 4d quaternion vector

    This algorithm is based on algorithm described in
    https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L201

    Args:
        rotation_matrix (Tensor): the rotation matrix to convert.

    Return:
        Tensor: the rotation in quaternion

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 4)`

    Example:
        >>> input = torch.rand(4, 3, 4)  # Nx3x4
        >>> output = tgm.rotation_matrix_to_quaternion(input)  # Nx4
    """
    rmat_t = ops.transpose(rotation_matrix, (0, 2, 1))

    mask_d2 = rmat_t[:, 2, 2] < eps
    mask_d2 = mask_d2.astype(ms.int64)

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]
    mask_d0_d1 = mask_d0_d1.astype(ms.int64)
    mask_d0_nd1 = mask_d0_nd1.astype(ms.int64)

    t_0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q_0 = ops.stack(
        (
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            t_0,
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
        ),
        axis=-1,
    )
    t0_rep = ops.transpose(ms.numpy.tile(t_0, (4, 1)), (1, 0))

    t_1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q_1 = ops.stack(
        (
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
            t_1,
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
        ),
        axis=-1,
    )
    t1_rep = ops.transpose(ms.numpy.tile(t_1, (4, 1)), (1, 0))

    t_2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q_2 = ops.stack(
        (
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
            rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
            rmat_t[:, 1, 2] + rmat_t[:, 2, 1],
            t_2,
        ),
        axis=-1,
    )
    t2_rep = ops.transpose(ms.numpy.tile(t_2, (4, 1)), (1, 0))

    t_3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q_3 = ops.stack(
        (
            t_3,
            rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
            rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
            rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
        ),
        axis=-1,
    )
    t3_rep = ops.transpose(ms.numpy.tile(t_3, (4, 1)), (1, 0))

    mask_c0 = (mask_d2 * mask_d0_d1).astype(ms.bool_)
    anti_mask_d0_d1 = mask_d0_d1.astype(ms.bool_)
    anti_mask_d0_d1 = ~anti_mask_d0_d1
    anti_mask_d0_d1 = anti_mask_d0_d1.astype(ms.int64)
    anti_mask_d2 = mask_d2.astype(ms.bool_)
    anti_mask_d2 = ~anti_mask_d2
    anti_mask_d2 = anti_mask_d2.astype(ms.int64)
    mask_c1 = (mask_d2 * anti_mask_d0_d1).astype(ms.bool_)
    mask_c2 = (anti_mask_d2 * mask_d0_nd1).astype(ms.bool_)
    mask_c3 = (anti_mask_d2 * anti_mask_d0_d1).astype(ms.bool_)
    mask_c0 = mask_c0.view(-1, 1).astype(q_0.dtype)
    mask_c1 = mask_c1.view(-1, 1).astype(q_1.dtype)
    mask_c2 = mask_c2.view(-1, 1).astype(q_2.dtype)
    mask_c3 = mask_c3.view(-1, 1).astype(q_3.dtype)

    q_p = q_0 * mask_c0 + q_1 * mask_c1 + q_2 * mask_c2 + q_3 * mask_c3
    sqrt = ops.Sqrt()
    q_p /= sqrt(
        t0_rep * mask_c0
        + t1_rep * mask_c1
        + t2_rep * mask_c2  # noqa
        + t3_rep * mask_c3
    )  # noqa
    q_p *= 0.5
    return q_p


def rotation_matrix_to_angle_axis(rotation_matrix):
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert 3x4 rotation matrix to Rodrigues vector

    Args:
        rotation_matrix (Tensor): rotation matrix.

    Returns:
        Tensor: Rodrigues vector transformation.

    Shape:
        - Input: :math:`(N, 3, 4)`
        - Output: :math:`(N, 3)`

    Example:
        >>> input = torch.rand(2, 3, 4)  # Nx4x4
        >>> output = tgm.rotation_matrix_to_angle_axis(input)  # Nx3
    """
    if rotation_matrix.shape[1:] == (3, 3):
        rot_mat = rotation_matrix.reshape(-1, 3, 3)
        hom = broadcast_to(
            Tensor([0, 0, 1], dtype=ms.float32).reshape(1, 3, 1),
            (rot_mat.shape[0], -1, -1),
        )
        rotation_matrix = ops.concat((rot_mat, hom), axis=-1)

    quaternion = rotation_matrix_to_quaternion(rotation_matrix)
    a_a = quaternion_to_angle_axis(quaternion)
    a_a[ops.isnan(a_a)] = 0.0
    return a_a


def quaternion_to_angle_axis(quaternion: Tensor) -> Tensor:
    """
    This function is borrowed from https://github.com/kornia/kornia

    Convert quaternion vector to angle axis of rotation.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion (torch.Tensor): tensor with quaternions.

    Return:
        torch.Tensor: tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = tgm.quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not isinstance(quaternion, Tensor):
        raise TypeError(
            "Input type is not a torch.Tensor. Got {}".format(type(quaternion))
        )

    if not quaternion.shape[-1] == 4:
        raise ValueError(
            "Input must be a tensor of shape Nx4 or 4. Got {}".format(quaternion.shape)
        )
    # unpack input and compute conversion
    q_1: Tensor = quaternion[..., 1]
    q_2: Tensor = quaternion[..., 2]
    q_3: Tensor = quaternion[..., 3]
    sin_squared_theta: Tensor = q_1 * q_1 + q_2 * q_2 + q_3 * q_3

    sqrt = ops.Sqrt()
    sin_theta: Tensor = sqrt(sin_squared_theta)
    cos_theta: Tensor = quaternion[..., 0]
    two_theta: Tensor = 2.0 * numpy.where(
        cos_theta < 0.0,
        ops.atan2(-sin_theta, -cos_theta),
        ops.atan2(sin_theta, cos_theta),
    )

    k_pos: Tensor = two_theta / sin_theta
    zeroslike = ops.ZerosLike()
    k_neg: Tensor = 2.0 * zeroslike(sin_theta)
    k: Tensor = numpy.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis: Tensor = zeroslike(quaternion)[..., :3]
    angle_axis[..., 0] += q_1 * k
    angle_axis[..., 1] += q_2 * k
    angle_axis[..., 2] += q_3 * k
    return angle_axis
