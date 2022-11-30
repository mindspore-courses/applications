"""smpl lib"""
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
from typing import Tuple, Optional
import mindspore as ms
import mindspore.nn as nn
from mindspore import numpy as np
from mindspore.ops.function import broadcast_to

from .utils import Tensor


def lbs(
        betas: Tensor,
        pose: Tensor,
        v_template: Tensor,
        shapedirs: Tensor,
        posedirs: Tensor,
        j_regressor: Tensor,
        parents: Tensor,
        lbs_weights: Tensor,
        pose2rot: bool = True,
        return_transformation: bool = False,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Optional[Tensor]]:
    """ Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : mindspore.tensor BxNB
            The tensor of shape parameters
        pose : mindspore.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template mindspore.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : mindspore.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : mindspore.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : mindspore.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: mindspore.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: mindspore.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex
        pose2rot: bool, optional
            Flag on whether to convert the input pose tensor to rotation
            matrices. The default value is True. If False, then the pose tensor
            should already contain rotation matrices and have a size of
            Bx(J + 1)x9
        dtype: mindspore.dtype, optional

        Returns
        -------
        verts: mindspore.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: mindspore.tensor BxJx3
            The joints of the model
    """

    batch_size = max(betas.shape[0], pose.shape[0])

    dtype = betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    j_p = vertices2joints(j_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = ms.ops.eye(3, 3, dtype)
    if pose2rot:
        rot_mats = batch_rodrigues(pose.view(-1, 3)).view([batch_size, -1, 3, 3])

        pose_feature = (rot_mats[:, 1:, :, :] - ident).view([batch_size, -1])
        # (N x P) x (P, V * 3) -> N x V x 3
        pose_offsets = ms.ops.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    else:
        pose_feature = pose[:, 1:].view(batch_size, -1, 3, 3) - ident
        rot_mats = pose.view(batch_size, -1, 3, 3)

        pose_offsets = ms.ops.matmul(pose_feature.view(batch_size, -1), posedirs).view(
            batch_size, -1, 3
        )

    v_posed = pose_offsets + v_shaped
    # 4. Get the global joint location
    j_transformed, a_p = batch_rigid_transform(rot_mats, j_p, parents)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    expand_dims = ms.ops.ExpandDims()
    w_p = broadcast_to(expand_dims(lbs_weights, 0), (batch_size, -1, -1))
    # W = Tensor(numpy.load("w.npy"))
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = j_regressor.shape[0]

    t_p = ms.ops.matmul(w_p, a_p.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    ones = ms.ops.Ones()
    homogen_coord = ones((batch_size, v_posed.shape[1], 1), dtype)
    cat = ms.ops.Concat(axis=2)
    v_posed_homo = cat([v_posed, homogen_coord])
    v_homo = ms.ops.matmul(t_p, expand_dims(v_posed_homo, -1))

    verts = v_homo[:, :, :3, 0]

    if return_transformation:
        return verts, j_transformed, a_p, t_p

    return verts, j_transformed


def blend_shapes(betas: Tensor, shape_disps: Tensor) -> Tensor:
    """ Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : mindspore.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: mindspore.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    mindspore.tensor BxVx3
        The per-vertex displacement due to shape deformation
    """

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    einsum = ms.ops.Einsum("bl,mkl->bmk")
    blend_shape = einsum((betas, shape_disps))
    return blend_shape


def vertices2joints(j_regressor: Tensor, vertices: Tensor) -> Tensor:
    """ Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : mindspore.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : mindspore.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    mindspore.tensor BxJx3
        The location of the joints
    """
    einsum = ms.ops.Einsum("bik,ji->bjk")
    return einsum((vertices, j_regressor))


def batch_rodrigues(rot_vecs: Tensor) -> Tensor:
    """ Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: mindspore.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: mindspore.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    """

    batch_size = rot_vecs.shape[0]
    dtype = rot_vecs.dtype

    angle = ms.ops.norm(rot_vecs + 1e-8, keep_dims=True, axis=1)
    rot_dir = rot_vecs / angle

    expand_dims = ms.ops.ExpandDims()
    cos_op = ms.ops.Cos()
    sin_op = ms.ops.Sin()
    cos = expand_dims(cos_op(angle), 1)
    sin = expand_dims(sin_op(angle), 1)

    # Bx1 arrays
    r_x, r_y, r_z = ms.ops.split(input_x=rot_dir, output_num=1, axis=1)
    zeros = ms.ops.Zeros()
    k_p = zeros((batch_size, 3, 3), dtype)

    zeros = zeros((batch_size, 1), dtype)
    cat = ms.ops.Concat()
    k_p = cat([zeros, -r_z, r_y, r_z, zeros, -r_x, -r_y, r_z, zeros], 1).view(
        (batch_size, 3, 3)
    )

    eye = ms.ops.Eye()
    ident = expand_dims(eye(3, 3, dtype), 0)
    batmatmul = ms.ops.BatchMatMul()
    rot_mat = ident + sin * k_p + (1 - cos) * batmatmul(k_p, k_p)
    return rot_mat


def batch_rigid_transform(
        rot_mats: Tensor, joints: Tensor, parents: Tensor
) -> Tensor:
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : mindspore.tensor BxNx3x3
        Tensor of rotation matrices
    joints : mindspore.tensor BxNx3
        Locations of joints
    parents : mindspore.tensor BxN
        The kinematic tree of each object
    dtype : mindspore.dtype, optional:
        The mindspore type of the created tensors, the default is mindspore.float32

    Returns
    -------
    posed_joints : mindspore.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : mindspore.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints
    """

    # joints = Tensor(numpy.load("joints.npy"))
    # rot_mats = Tensor(numpy.load("rot_mats.npy"))

    joints = ms.ops.expand_dims(joints, axis=-1)

    rel_joints = joints.copy()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    rel_joints = ms.ops.reshape(rel_joints, (-1, 3, 1))

    transforms_mat = ms.ops.reshape(
        transform_mat(ms.ops.reshape(rot_mats, (-1, 3, 3)), rel_joints),
        (-1, joints.shape[1], 4, 4),
    )

    # transforms_mat = Tensor(numpy.load("transforms_mat.npy"))

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = ms.ops.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    stack = ms.ops.Stack(axis=1)
    transforms = stack(transform_chain)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    pad_op1 = nn.Pad(paddings=((0, 0), (0, 0), (0, 1), (0, 0)))
    joints_homogen = pad_op1(joints)

    pad_op2 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (3, 0)))

    rel_transforms = transforms - pad_op2(ms.ops.matmul(transforms, joints_homogen))

    return posed_joints, rel_transforms


def transform_mat(r_p: Tensor, t_p: Tensor) -> Tensor:
    """ Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    """
    # No padding left or right, only add an extra row

    return ms.ops.concat(
        [
            np.pad(r_p, ((0, 0), (0, 1), (0, 0))),
            np.pad(t_p, ((0, 0), (0, 1), (0, 0)), constant_values=1),
        ],
        axis=2,
    )
