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
"""matcher"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
import mindspore as ms
from mindspore.ops import stop_gradient
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.tensor import Tensor


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding box coordinates from xywh to cxcyhw.

    Args:
        x (Tensor): Bounding box coordinates with shape [num, 4].

    Returns:
        Tenosr: Bounding box coordinates with shape [num, 4].
    """
    unstack = ops.Unstack(-1)
    x_c, y_c, w, h = unstack(x)
    box = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    stack = ops.Stack(-1)
    return stack(box)


def box_iou(boxes1, boxes2):
    """
    Calculate the iou of two boxes

    Args:
        boxes1 (Tensor): Bounding box coordinates with shape [N, 4].
        boxes2 (Tensor): Bounding box coordinates with shape [M, 4].

    Returns:
        iou (Tensor): iou of two boxes with shape [N, M]
        union (Tensor): union of two boxes with shape [N, M].
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    maximum = ops.Maximum()
    minimum = ops.Minimum()
    lt = maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = ops.clip_by_value((rb - lt), 0, 10000)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Args:
        boxes1 (Tensor): Bounding box coordinates with shape [N, 4].
        boxes2 (Tensor): Bounding box coordinates with shape [M, 4].

    Returns:
        Tensor: Giou of two boxes with shape [N, M]
    """
    iou, union = box_iou(boxes1, boxes2)
    maximum = ops.Maximum()
    minimum = ops.Minimum()
    lt = minimum(boxes1[:, None, :2], boxes2[:, :2])
    rb = maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = ops.clip_by_value((rb - lt), 0, 10000)
    area = wh[:, :, 0] * wh[:, :, 1]
    return iou - (area - union) / area


def cross_entropy(logits, labels, weight):
    """
    Cross entropy function.

    Args:
        logits (Tensor): The classification logits with shape [batch_size, num_queries, num_classes].
        labels (Tensor): The classification labels with shape [batch_size, num_queries].
        weight (Tensor): The weight for nll_loss with shape [num_classes,].

    Returns:
        Tensor: Cross entropy loss
    """
    softmax = nn.Softmax(1)
    log = ops.Log()
    nll_loss = ops.NLLLoss()
    bs = logits.shape[0]
    loss = 0
    for i in range(bs):
        soft_out = softmax(logits[i])
        log_soft_out = log(soft_out)
        loss += nll_loss(log_soft_out, labels[i], weight)[0]
    return loss / bs


def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k.

    Args:
        output (Tensor): The classification logits with shape [num_idx, num_classes].
        target (Tensor): The classification labels ids with shape [num_idx,].
        topk (tuple): The top k.

    Returns:
        list: Accuracy result.
    """
    if ops.Size()(target) == 0:
        return [Tensor(0.0)]
    maxk = max(topk)
    batch_size = target.shape[0]
    _, pred = ops.TopK(sorted=True)(output, maxk)
    pred = pred.T
    correct = ops.Equal()(pred, target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).sum(0)
        res.append(stop_gradient(ops.Mul()(correct_k, 100.0 / batch_size)))
    return res


class HungarianMatcher(nn.Cell):
    """
    This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).

    Args:
        cost_class (float): The relative weight of the classification error in the matching cost.
        cost_bbox (float): The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        cost_giou (float): The relative weight of the giou loss of the bounding box in the matching cost.

    Inputs:
        - **outputs** (dict) - This is a dict that contains at least these entries:
                 "pred_logits": [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": [batch_size, num_queries, 4] with the predicted box coordinates.
        - **targets** (list) - This is a list of targets (len(targets) = batch_size),
                 where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

    Outputs:
        A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> hm = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)
        >>> indices = hm(output, target)
    """

    def __init__(self, cost_class=1.0, cost_bbox=1.0, cost_giou=1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def construct(self, outputs, targets):
        """ Apply HungarianMatcher"""
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].view((bs * num_queries, -1))
        out_prob = nn.Softmax()(out_prob)
        out_bbox = outputs["pred_boxes"].view((bs * num_queries, 4))
        concat = ops.Concat()
        tgt_ids = concat([v["labels"] for v in targets])
        tgt_bbox = concat([v["boxes"] for v in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = distance.cdist(out_bbox.asnumpy(), tgt_bbox.asnumpy(), 'cityblock')
        cost_bbox = Tensor(cost_bbox)
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost = cost.view(bs, num_queries, -1)
        sizes = []
        for v in targets:
            sizes.append(len(v["boxes"]))
        c_split = []
        begin = 0
        for i in sizes:
            c_split.append(cost[:, :, begin:begin + i])
            begin += i
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(c_split)]
        return [(stop_gradient(Tensor(i)), stop_gradient(Tensor(j))) for i, j in indices]


def _max_by_axis(the_list):
    """
    Get the maximum width and height in the batch images.

    Args:
        the_list (list): A list of inputs shape.

    Returns:
        list: Maximum inputs shape.
    """
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def batch_fn(masks):
    """
    Padding masks so that all masks are the same size.

    Args:
        masks (list): A list of mask.

    Returns:
        Tensor: Batch masks.
    """
    max_size = _max_by_axis([list(mask.shape) for mask in masks])
    batch_shape = [len(masks)] + max_size
    bs = batch_shape[0]
    mask_p = Tensor(np.zeros(batch_shape).astype(np.float32))
    for i in range(bs):
        mask_p[i][: masks[i].shape[0], : masks[i].shape[1], : masks[i].shape[2]] = masks[i]
    return mask_p


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs (Tensor): A float tensor of arbitrary shape. The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha (Tensor): (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma (Tensor): Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor.
    """
    sigmoid = ops.Sigmoid()
    prob = sigmoid(inputs)
    binary_cross_entropy = ops.BinaryCrossEntropy(reduction="none")
    weight = Tensor(np.ones((inputs.shape)).astype(np.float32))
    inputs = sigmoid(inputs)
    targets = sigmoid(targets)
    ce_loss = binary_cross_entropy(inputs, targets, weight)
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.mean(1).sum() / num_boxes


def dice_loss(inputs, targets, num_boxes):
    """
    Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs (Tensor): A float tensor of arbitrary shape.
                The predictions for each example.
        targets (Tensor): A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).

    Return:
        float, dice loss.
    """
    sigmoid = ops.Sigmoid()
    inputs = sigmoid(inputs)
    inputs = inputs.view((inputs.shape[0], -1))
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


class SetCriterion(nn.Cell):
    """
    This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)

    Args:
        num_classes (int): Number of object categories, omitting the special no-object category.
        matcher (Cell): Module able to compute a matching between targets and proposals.
        weight_dict (dict): Dict containing as key the names of the losses and as values their relative weight.
        eos_coef (float): Relative classification weight applied to the no-object category.
        losses (dict): list of all the losses to be applied.

    Inputs:
        - **outputs** (dict) - This is a dict that contains at least these entries:
                 "pred_logits": [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": [batch_size, num_queries, 4] with the predicted box coordinates.
        - **targets** (list) - This is a list of targets (len(targets) = batch_size),
                 where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

    Outputs:
        A dict containing as key the names of the losses and as values their weight.

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> hm = HungarianMatcher(cost_class=1, cost_bbox=1, cost_giou=1)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        ones = ops.Ones()
        self.empty_weight = ones(self.num_classes + 1, ms.float32)
        self.empty_weight[-1] = self.eos_coef
        self.empty_weight.requires_grad = False

    def loss_labels(self, outputs, targets, indices, _):
        """
        Classification loss (NLL)

        Args:
            outputs (dict): This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets (list): This is a list of targets (len(targets) = batch_size),
                 where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            indices (tuple): tuples of (batch_idx, src_idx) where:
                batch_idx is the indices of the selected predictions (in order)
                src_idx is the indices of the corresponding selected targets (in order)

        Returns:
            dict, Classification loss and class error.
        """
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        concat = ops.Concat()
        target_classes_o = concat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes = ms.numpy.full(src_logits.shape[:2], self.num_classes)
        target_classes[idx] = target_classes_o
        loss_ce = cross_entropy(src_logits, target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    def loss_cardinality(self, outputs, targets, *args):
        """
        Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.

        Args:
            outputs (dict): This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets (list): This is a list of targets (len(targets) = batch_size),
                 where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            dict, cardinality error.

        """
        if args is None:
            return {'cardinality_error': stop_gradient(0)}
        pred_logits = outputs['pred_logits']
        tgt_lengths = Tensor([len(v["labels"]) for v in targets])
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        l1_loss = nn.L1Loss()
        card_err = l1_loss(card_pred, tgt_lengths)
        losses = {'cardinality_error': stop_gradient(card_err)}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss

        Args:
            outputs (dict): This is a dict that contains at least these entries:
                    "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                    "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets (list): This is a list of targets (len(targets) = batch_size),
                    where each target is a dict containing:
                    "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                            objects in the target) containing the class labels
                    "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            indices (tuple): tuples of (batch_idx, src_idx) where:
                batch_idx is the indices of the selected predictions (in order)
                src_idx is the indices of the corresponding selected targets (in order)
            num_boxes (int): The number of bounding box.

        Return:
            dict, bounding box loss and giou loss.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = ops.Concat(0)([t['boxes'][i] for t, (_, i) in zip(targets, indices)])
        l1_loss = nn.L1Loss(reduction='none')
        loss_bbox = l1_loss(src_boxes, target_boxes)
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - ms.numpy.diag(generalized_box_iou(box_cxcywh_to_xyxy(src_boxes),
                                                          box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]

        Args:
            outputs (dict): This is a dict that contains at least these entries:
                    "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                    "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets (list): This is a list of targets (len(targets) = batch_size),
                    where each target is a dict containing:
                    "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                            objects in the target) containing the class labels
                    "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                    "masks: containing a tensor of dim [nb_target_boxes, h, w]
            indices (tuple): tuples of (batch_idx, src_idx) where:
                batch_idx is the indices of the selected predictions (in order)
                src_idx is the indices of the corresponding selected targets (in order)
            num_boxes (int): The number of bounding box.

        Return:
            dict, the focal loss and the dice loss.
        """
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        target_masks = batch_fn(masks)
        target_masks = target_masks[tgt_idx]
        resize = ops.ResizeNearestNeighbor(target_masks.shape[-2:])
        src_masks = resize(src_masks[:, None])
        flatten = ops.Flatten()
        src_masks = flatten(src_masks[:, 0])
        target_masks = flatten(target_masks)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        """
        permute predictions following indices

        Args:
            indices (tuple): tuples of (batch_idx, src_idx) where:
                batch_idx is the indices of the selected predictions (in order)
                src_idx is the indices of the corresponding selected targets (in order)

        Returns:
            batch_idx (Tensor): Tensor with shape [batch_idx, ].
            src_idx (Tensor): Tensor with shape [batch_idx, ].
        """
        concat = ops.Concat()
        batch_idx = concat([ms.numpy.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = concat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """
        permute predictions following indices

        Args:
            indices (tuple): tuples of (batch_idx, tgt_idx) where:
                batch_idx is the indices of the selected predictions (in order)
                tgt_idx is the indices of the corresponding selected targets (in order)

        Returns:
            batch_idx (Tensor): Tensor with shape [batch_idx, ].
            tgt_idx (Tensor): Tensor with shape [batch_idx, ].
        """
        concat = ops.Concat()
        batch_idx = concat([ms.numpy.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = concat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        """
        Get loss.

        Args:
            outputs (dict): This is a dict that contains at least these entries:
                    "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                    "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets (list): This is a list of targets (len(targets) = batch_size),
                    where each target is a dict containing:
                    "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                            objects in the target) containing the class labels
                    "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
                    "masks: containing a tensor of dim [nb_target_boxes, h, w]
            indices (tuple): tuples of (batch_idx, src_idx) where:
                batch_idx is the indices of the selected predictions (in order)
                src_idx is the indices of the corresponding selected targets (in order)
            num_boxes (int): The number of bounding box.

        Return:
            dict, loss of loss_map.
        """
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def construct(self, outputs, targets):
        """ Apply SetCriterion. """
        outputs_without_aux = {}
        for k, v in outputs.items():
            if k != 'aux_outputs':
                outputs_without_aux[k] = v
        indices = self.matcher(outputs_without_aux, targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = Tensor([num_boxes])
        num_boxes = ops.clip_by_value(num_boxes, 1, 10000).item()
        losses = {}
        for loss in self.losses:
            tmp_loss = self.get_loss(loss, outputs, targets, indices, num_boxes)
            for k, v in tmp_loss.items():
                losses[k] = v
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        continue
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    for k, v in l_dict.items():
                        losses[k + f'_{i}'] = v
        return losses


def build_matcher(cost_class=1, cost_bbox=5, cost_giou=2):
    """
    Build Hungarian matcher

    Args:
        cost_class (float): The relative weight of the classification error in the matching cost.
        cost_bbox (float): The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        cost_giou (float): The relative weight of the giou loss of the bounding box in the matching cost.

    Returns:
        Cell: Cell of HungarianMatcher.
    """
    return HungarianMatcher(cost_class=cost_class, cost_bbox=cost_bbox, cost_giou=cost_giou)


def build_criterion(is_segmentation=False):
    """
    Build loss function class.

    Args:
        is_segmentation (bool, optional): loss for segmentation if true. Defaults to False.

    Returns:
        Cell: Cell of criterion.
    """
    num_classes = 91
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes', 'cardinality']
    if is_segmentation:
        num_classes = 250
        losses += ["masks"]
        weight_dict["loss_mask"] = 1
        weight_dict["loss_dice"] = 1
    hm = build_matcher()
    criterion = SetCriterion(num_classes, matcher=hm, weight_dict=weight_dict, eos_coef=0.1, losses=losses)
    return criterion
