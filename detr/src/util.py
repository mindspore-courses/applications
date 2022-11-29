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
# ======================================================================
"""util code"""

import cv2
import numpy as np
from panopticapi.utils import id2rgb, rgb2id
import mindspore
from mindspore.common.tensor import Tensor
from mindspore import ops


def box_cxcywh_to_xywh(x):
    """
    Convert bounding box coordinates from xywh to cxcyhw.

    Args:
        x (Tensor): Bounding box coordinates with shape [num, 4].

    Returns:
        Tenosr, bounding box coordinates with shape [num, 4].
    """
    unstack = ops.Unstack(-1)
    x_c, y_c, w, h = unstack(x)
    box = [(x_c - 0.5 * w), (y_c - 0.5 * h), w, h]
    stack = ops.Stack(-1)
    return stack(box)


def post_process(outputs, target_sizes):
    """
    Perform the computation

    Args:
        outputs (dict): This is a dict that contains at least these entries:
                 "pred_logits": [batch_size, num_queries, num_classes] with the classification logits.
                 "pred_boxes": [batch_size, num_queries, 4] with the predicted box coordinates.
        target_sizes (Tensor): The original size of images with shape [batch_size, 2].

    Return:
        list, a list of detection result of each image.
    """
    out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
    softmax = ops.Softmax(-1)
    prob = softmax(out_logits)
    scores = prob[..., :-1].max(-1)
    labels = prob[..., :-1].argmax(-1)
    boxes = box_cxcywh_to_xywh(out_bbox)
    unstack = ops.Unstack(1)
    img_h, img_w = unstack(target_sizes)
    stack = ops.Stack(1)
    scale_fct = stack([img_w, img_h, img_w, img_h])
    boxes = boxes * scale_fct[:, None, :]
    results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
    return results


def post_process_seg(results, outputs, orig_target_sizes, max_target_sizes):
    """
    According to the output, the segmentation result is obtained.

    Args:
        results (list): A list of detection result of each image.
        outputs (dict): This is a dict that contains at least these entries:
                 "pred_logits": [batch_size, num_queries, num_classes] with the classification logits.
                 "pred_boxes": [batch_size, num_queries, 4] with the predicted box coordinates.
                 "pred_masks": [batch_size, num_queries, H, W] with the masks for all queries.
        orig_target_sizes (Tensor): The original size of images with shape [batch_size, 2].
        max_target_sizes (Tensor): The maximum size of images with shape [batch_size, 2].

    Returns:
        list, a list of detection result of each image.
    """
    max_size = max_target_sizes.astype('float32').max(0)
    max_h = int(max_size[0])
    max_w = int(max_size[1])
    outputs_masks = outputs["pred_masks"]
    resize_bilinear = ops.ResizeBilinear((max_h, max_w))
    outputs_masks = resize_bilinear(outputs_masks)
    sigmoid = ops.Sigmoid()
    outputs_masks = sigmoid(outputs_masks) > 0.5
    for i, (cur_mask, t, tt) in enumerate(zip(outputs_masks, max_target_sizes, orig_target_sizes)):
        img_h, img_w = t[0], t[1]
        results[i]["masks"] = cur_mask[:, :img_h, :img_w]
        tt_h = int(tt[0])
        tt_w = int(tt[1])
        resize = ops.ResizeNearestNeighbor((tt_h, tt_w))
        ex_shape = (1,) + results[i]["masks"].shape
        results[i]["masks"] = resize(results[i]["masks"].astype(mindspore.float32).view(ex_shape))[0]
    return results


def post_process_pano(outputs, processed_sizes, target_sizes):
    """
    According to the output, the panorama segmentation result is obtained.

    Args:
        outputs (dict): This is a dict that contains at least these entries:
                 "pred_logits": [batch_size, num_queries, num_classes] with the classification logits.
                 "pred_boxes": [batch_size, num_queries, 4] with the predicted box coordinates.
        processed_sizes (Tensor): The size after data augmentation but before batching. Shape [batch_size, 2].
        target_sizes (Tensor): The original size of images with shape [batch_size, 2].
    Return:
        list, a list of segmentation result of each image.
    """
    out_logits, raw_masks = outputs["pred_logits"], outputs["pred_masks"]
    preds = []
    for cur_logits, cur_masks, size, target_size in zip(out_logits, raw_masks, processed_sizes, target_sizes):
        softmax = ops.Softmax()
        cur_logits_softmax = softmax(cur_logits)
        scores = cur_logits_softmax.max(-1)
        labels = cur_logits_softmax.argmax(-1)
        labels_id = labels != (outputs["pred_logits"].shape[-1] - 1)
        scores_id = scores > 0.85
        keep = []
        for idx, (lab_id, sco_id) in enumerate(zip(labels_id, scores_id)):
            if lab_id and sco_id:
                keep.append(idx)
        if not keep:
            h, w = cur_masks.shape[-2:]
            m_id = Tensor(np.zeros((h, w)).astype(np.float32))
            seg_img = id2rgb(m_id.view(h, w).asnumpy())
            final_h, final_w = int(target_size[0]), int(target_size[1])
            seg_img = cv2.resize(seg_img, (final_w, final_h), interpolation=cv2.INTER_NEAREST)
            b, g, r = cv2.split(seg_img)
            seg_img = cv2.merge([r, g, b])
            predictions = {"segments_info": [], "file_name": seg_img}
            preds.append(predictions)
            continue
        cur_scores = cur_logits_softmax.max(-1)
        cur_classes = cur_logits_softmax.argmax(-1)
        cur_scores = cur_scores[keep]
        cur_classes = cur_classes[keep]
        cur_masks = cur_masks[keep]
        resize_bilinear = ops.ResizeBilinear((int(size[0]), int(size[1])))
        cur_masks = resize_bilinear(cur_masks[None, :])[0]
        h, w = cur_masks.shape[-2:]
        flatten = ops.Flatten()
        cur_masks = flatten(cur_masks)
        stuff_equiv_classes = {}
        for k, label in enumerate(cur_classes):
            if not label < 91:
                if int(label) not in stuff_equiv_classes:
                    stuff_equiv_classes[int(label)] = [k]
                else:
                    stuff_equiv_classes[int(label)].append(k)
        area, seg_img = get_ids_area(cur_masks, cur_scores, h, w, stuff_equiv_classes, target_size, dedup=True)
        if cur_classes.size > 0:
            while True:
                filtered_not_small = []
                for i, a in enumerate(area):
                    if a > 4:
                        filtered_not_small.append(i)
                if len(filtered_not_small) < len(area):
                    cur_scores = cur_scores[filtered_not_small]
                    cur_classes = cur_classes[filtered_not_small]
                    cur_masks = cur_masks[filtered_not_small]
                    area, seg_img = get_ids_area(cur_masks, cur_scores, h, w, stuff_equiv_classes, target_size)
                else:
                    break
        else:
            cur_classes = Tensor(np.ones(1).astype(np.float32))
        segments_info = []
        for i, a in enumerate(area):
            cat = cur_classes[i]
            segments_info.append({"id": i, "isthing": int(cat) < 91, "category_id": int(cat), "area": int(a)})
        del cur_classes
        b, g, r = cv2.split(seg_img)
        seg_img = cv2.merge([r, g, b])
        predictions = {"segments_info": segments_info, "file_name": seg_img}
        preds.append(predictions)
    return preds


def get_ids_area(masks, scores, height, width, stuff_equiv_classes, target_size, dedup=False):
    """
    This helper function creates the final panoptic segmentation image.
    It also returns the area of the masks that appears on the image.

    Args:
        masks (Tensor):  Mask of images with shape [num, H*W)
        scores (Tenor): Scores of images mask with shape [num]
        height (int): Height of masks.
        width (int): Width of masks.
        stuff_equiv_classes (dict): _description_
        target_size (Tensor): The original size of each image.
        dedup (bool, optional): Merge the masks corresponding to the
                                same stuff class if true. Defaults to False.

    Returns:
        area (list), a list of masks area.
        seg_img (Tensor), segmentation image with the size of target_size.
    """
    transpose = ops.Transpose()
    m_id = ops.Softmax()(transpose(masks, (1, 0)))
    if m_id.shape[-1] == 0:
        m_id = Tensor(np.zeros((height, width)).astype(np.float32))
    else:
        m_id = m_id.argmax(-1).view(height, width)
    if dedup:
        for equiv in stuff_equiv_classes.values():
            if len(equiv) > 1:
                for eq_id in equiv:
                    m_id = ops.MaskedFill()(m_id, m_id == eq_id, Tensor(equiv[0]).astype('int32'))
    final_h, final_w = int(target_size[0]), int(target_size[1])
    seg_img = id2rgb(m_id.view(height, width).asnumpy())
    seg_img = cv2.resize(seg_img, (final_w, final_h), interpolation=cv2.INTER_NEAREST)
    np_seg_img = seg_img.reshape(final_h, final_w, 3)
    m_id = rgb2id(np_seg_img)
    area = []
    for i in range(len(scores)):
        area.append((m_id == i).sum())
    return area, seg_img
