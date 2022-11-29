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
# ==============================================================================
"""Some Transform Functions for Keypoints"""
import cv2
import numpy as np


def get_affine_transform(center, scale, rot, output_size):
    """
    Get Affine Transform Matrix for Keypoints Image

    Args:
        center (np.ndarray): Image Center Point
        scale (np.ndarray): Image Pixel Scale
        rot (float): Rotation Degree
        output_size (list): Output Shape

    Returns:
        Specific Affine Transform Matrix

    Examples:
        >>> trans = get_affine_transform(np.array([1.5, 1.0]), np.array([0.1, 0.2]), 90.0, [128, 128])
    """
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale * 200.0

    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)  # Three source points for affine transform
    dst = np.zeros((3, 2), dtype=np.float32)  # Three destination points for affine transform
    src[0, :] = center
    src[1, :] = center + src_dir
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """
    Get Affine Transform for Keypoints

    Args:
        pt (np.ndarray): Keypoints List
        t: Transform Method

    Returns:
        Affine Transform Result List for Keypoints

    Examples:
        >>> trans = get_affine_transform(np.array([1.5, 1.0]), np.array([0.1, 0.2]), 90.0, [128, 128])
        >>> joints = affine_transform(np.array([1.3, 1.5]), trans)
    """
    new_pt = np.array([pt[0], pt[1], 1.])
    new_pt = np.dot(t, new_pt)

    return new_pt[:2]


def get_3rd_point(a, b):
    """
    Get 3rd Point of Affine Matrix

    Args:
        a (np.ndarray): 1st Point
        b (np.ndarray): 2nd Point

    Returns:
        3rd point for affine matrix

    Examples:
        >>> src = get_3rd_point(np.array([1.5, 1.6]), np.array([1.4, 1.7]))
    """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """
    Get Rotation Direction for Affine Matrix Src Point

    Args:
        src_point (list): Src point for affine matrix
        rot_rad (float): Rotation Degree

    Returns:
        Src Point Rotation Direction

    Examples:
        >>> src_dir = get_dir([0, -64], 90.0)
    """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def flip_back(output, pairs):
    """
    Flip Model Output

    Args:
        output (np.ndarray): MSPN Output
        pairs (list): Keypoint Flip Pairs

    Returns:
        Flipped Array

    Examples:
        >>> outputs_flipped = flip_back(np.random.rand(2, 17, 64, 48), [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], \
        [11, 12], [13, 14], [15, 16]])
    """
    output = output[:, :, :, ::-1]

    for pair in pairs:
        tmp = output[:, pair[0], :, :].copy()
        output[:, pair[0], :, :] = output[:, pair[1], :, :]
        output[:, pair[1], :, :] = tmp

    return output


def flip_joints(joints, joints_vis, width, pairs):
    """
    Flip Joints

    Args:
        joints (np.ndarray): Joints List
        joints_vis (np.ndarray): Joints Visible List
        width (int): Image Width
        pairs (list): Keypoint Flip Pairs

    Returns:
        Flipped Joints and Joints Visible List

    Examples:
        >>> joints, joints_vis = flip_joints(np.array([[1.5, 1.6]]), np.array([[1]]), 128, [[1, 2], [3, 4], [5, 6], \
        [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]])
    """
    joints[:, 0] = width - joints[:, 0] - 1

    for pair in pairs:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints, joints_vis
