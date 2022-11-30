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

"""Some function about extracting and transforming predictions from heatmaps."""

import numpy as np
import cv2


def rescore(overlap, scores, thresh, net_type='gaussian'):
    """
    Rescoring overlapped areas in a heatmap.

    Args:
        overlap (numpy.ndarray): Overlapped areas.
        scores (numpy.ndarray): Heatmap score.
        thresh (float): The thresh to judge whether need rescoring.
        net_type (str): Network type, gaussian or linear. Default: gaussian.

    Return:
        numpy.ndarray, rescored score.
    """

    if net_type == 'linear':
        inds = np.where(overlap >= thresh)[0]
        scores[inds] = scores[inds] * (1 - overlap[inds])
    else:
        scores = scores * np.exp(- overlap**2 / thresh)

    return scores

def get_3rd_point(a, b):
    """
    Get 3rd point in the line defined by a and b.

    Args:
        a (numpy.ndarray): First point that defined the coord of the 3rd point.
        b (numpy.ndarray): Second points that defined the coord of the 3rd point.

    Returns:
        numpy.ndarray, the 3rd point.
    """

    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)

def get_dir(src_point, rot_rad):
    """
    Get a rotated point.

    Args:
        src_point (numpy.ndarray): Source point.
        rot_rad (float): Rotation angle.

    Return:
        numpy.ndarray, rotated point.
    """

    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def affine_transform(pt, t):
    """
    Get a new point from a given point and transform matrix.

    Args:
        pt (numpy.ndarray): Original point.
        t (numpy.ndarray): Transform matrix.

    Returns:
        numpy.ndarray, transformed point.
    """

    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def transform_preds(coords, center, scale, output_size):
    """
    Transform predicted keypoints to the original place in the input image.

    Args:
        center (numpy.ndarray): Center coords of the heatmap.
        scale (float or numpy.ndarray): Scale of the heatmap.
        rot (float): Rot angle of heatmap.
        shift (numpy.ndarray): Shift of heatmap.

    Returns:
        numpy.ndarray, transformed coords.
    """

    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    for p in range(coords.shape[0]):
        target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    return target_coords

def get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    """
    Get affine transform to linearly map heatmap.

    Args:
        center (numpy.ndarray): Center coords of the heatmap.
        scale (float or numpy.ndarray): Scale of the heatmap.
        rot (float): Rot angle of heatmap.
        shift (numpy.ndarray): Shift of heatmap.
        inv (bool): Get inverse transform if set to True.
        output_size (list): The size of output heatmap.

    Returns:
        numpy.ndarray, cv2 transform matrix.
    """

    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        print(scale)
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def get_max_preds(batch_heatmaps):
    """
    Get predictions from score maps by picking the coordinates with maximum value.

    Args:
        batch_heatmaps (numpy.ndarray): A batch of heatmaps.

    Return:
        numpy.ndarray, predicted joints coordinates.
        numpy.ndarray, heatmap values on the predicted joints coordinates.
    """

    batch_size = batch_heatmaps.shape[0]
    num_joints = batch_heatmaps.shape[1]
    width = batch_heatmaps.shape[3]
    heatmaps_reshaped = batch_heatmaps.reshape((batch_size, num_joints, -1))
    idx = np.argmax(heatmaps_reshaped, 2)
    maxvals = np.amax(heatmaps_reshaped, 2)

    maxvals = maxvals.reshape((batch_size, num_joints, 1))
    idx = idx.reshape((batch_size, num_joints, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

    preds[:, :, 0] = (preds[:, :, 0]) % width
    preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

    pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
    pred_mask = pred_mask.astype(np.float32)

    preds *= pred_mask
    return preds, maxvals


def get_final_preds(batch_heatmaps, center, scale):
    """
    Get prediction coordinates from heatmaps and transform them.

    Args:
        batch_heatmaps (numpy.ndarray): A batch of heatmaps.
        center (numpy.ndarray): Center coordinates of heatmaps.
        scale (numpy.ndarray): Scale of heatmaps.

    Return:
        numpy.ndarray, transformed predicted joints coordinates.
        numpy.ndarray, heatmap values on the predicted joints coordinates.
    """

    coords, maxvals = get_max_preds(batch_heatmaps)
    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]
    preds = coords.copy()

    # Transform back
    for i in range(coords.shape[0]):
        preds[i] = transform_preds(
            coords[i], center[i], scale[i], [heatmap_width, heatmap_height]
        )

    return preds, maxvals

def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    Flip joint coords.

    Args:
        joints (numpy.ndarray): Joints data.
        joint_vis (numpy.ndarray): Joints visibility.
        width (list): Image width.
        match_parts (list): A list that contain paired joint indices.

    Return:
        numpy.ndarray, flipped joints.
        numpy.ndarray, joints visibility.
    """

    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis
