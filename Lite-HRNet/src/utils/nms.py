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

"""NMS functions."""

import numpy as np


def nms(dets, thresh):
    """
    Greedily select boxes with high confidence and overlap with current maximum <= thresh,
    rule out overlap >= thresh.

    Args:
        dets (list): [[x1, y1, x2, y2 score],...].
        thresh (float): Retain overlap < thresh.

    Returns:
        list, indices to keep.
    """

    if dets.shape[0] == 0:
        return []

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def oks_iou(g, d, a_g, a_d, sigmas=None, in_vis_thre=None):
    """
    Compute oks iou between the current box and the maximum box

    Args:
        g (numpy.ndarray): The maximum box.
        d (numpy.ndarray): The current box.
        a_g (numpy.ndarray): The area of maximum box.
        a_d (numpy.ndarray):  The area of current box.
        sigmas (numpy.ndarray): The weight for each joints. Default: None.
        in_vis_thre (float): Visibility thre. Default: None.

    Returns:
        numpy.ndarray, iou score.
    """

    if not isinstance(sigmas, np.ndarray):
        sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72,
                           .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    variants = (sigmas * 2) ** 2
    xg = g[0::3]
    yg = g[1::3]
    vg = g[2::3]
    ious = np.zeros((d.shape[0]))
    for n_d in range(0, d.shape[0]):
        xd = d[n_d, 0::3]
        yd = d[n_d, 1::3]
        vd = d[n_d, 2::3]
        dx = xd - xg
        dy = yd - yg
        e = (dx ** 2 + dy ** 2) / variants / ((a_g + a_d[n_d]) / 2 + np.spacing(1)) / 2
        if in_vis_thre is not None:
            ind = list(vg > in_vis_thre) and list(vd > in_vis_thre)
            e = e[ind]
        ious[n_d] = np.sum(np.exp(-e)) / e.shape[0] if e.shape[0] != 0 else 0.0
    return ious

def oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    Greedily select boxes with high confidence and overlap with current maximum <= thresh,
    rule out overlap >= thresh, overlap = oks.

    Args:
        kpts_db (list): Keypoints data.
        thresh (float): Retain overlap < thresh.
        sigmas (numpy.ndarray): The weight for each joints. Default: None.
        in_vis_thre (float): Visibility thre. Default: None.

    Returns:
        list, indices to keep.
    """

    if not kpts_db:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        inds = np.where(oks_ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def soft_oks_nms(kpts_db, thresh, sigmas=None, in_vis_thre=None):
    """
    Greedily select boxes with high confidence and overlap with current maximum <= thresh,
    rule out overlap >= thresh, overlap = oks.

    Args:
        kpts_db (list): Keypoints data.
        thresh (float): Retain overlap < thresh.
        sigmas (numpy.ndarray): The weight for each joints. Default: None.
        in_vis_thre (float): Visibility thre. Default: None.

    Returns:
        list, indices to keep.
    """

    if not kpts_db:
        return []

    scores = np.array([kpts_db[i]['score'] for i in range(len(kpts_db))])
    kpts = np.array([kpts_db[i]['keypoints'].flatten() for i in range(len(kpts_db))])
    areas = np.array([kpts_db[i]['area'] for i in range(len(kpts_db))])

    order = scores.argsort()[::-1]
    scores = scores[order]

    max_dets = 20
    keep = np.zeros(max_dets, dtype=np.intp)
    keep_cnt = 0
    while order.size > 0 and keep_cnt < max_dets:
        i = order[0]

        oks_ovr = oks_iou(kpts[i], kpts[order[1:]], areas[i], areas[order[1:]], sigmas, in_vis_thre)

        order = order[1:]
        scores = rescore(oks_ovr, scores[1:], thresh)

        tmp = scores.argsort()[::-1]
        order = order[tmp]
        scores = scores[tmp]

        keep[keep_cnt] = i
        keep_cnt += 1

    keep = keep[:keep_cnt]

    return keep
