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
"""Repaired MultiBoxLoss."""

import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindvision.engine.loss.multiboxloss import SoftmaxCrossEntropyWithLogits


class MultiBoxLoss(nn.Cell):
    """"
    MultiBoxLoss for detection.

    Args:
        num_classes(int): Number of classes of objects to be detected.
        num_boxes(int): Number of priors per image.
        neg_pre_positive(int): The ratio of negative to positive in hard negative mining.
        batch_size(int): Training batch size.

    Returns:
        A tuple,which is (Box loss,Confidence loss,landmark loss).
    """

    def __init__(self, num_classes, num_boxes, neg_pre_positive, batch_size):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.neg_pre_positive = neg_pre_positive
        self.notequal = ops.NotEqual()
        self.less = ops.Less()
        self.tile = ops.Tile()
        self.reduce_sum = ops.ReduceSum()
        self.reduce_mean = ops.ReduceMean()
        self.expand_dims = ops.ExpandDims()
        self.smooth_l1_loss = ops.SmoothL1Loss()
        self.cross_entropy = SoftmaxCrossEntropyWithLogits()
        self.maximum = ops.Maximum()
        self.minimum = ops.Minimum()
        self.sort_descend = ops.TopK(True)
        self.sort = ops.TopK(True)
        self.gather = ops.GatherNd()
        self.max = ops.ReduceMax()
        self.log = ops.Log()
        self.exp = ops.Exp()
        self.concat = ops.Concat(axis=1)
        self.reduce_sum2 = ops.ReduceSum(keep_dims=True)
        self.idx = Tensor(np.reshape(np.arange(batch_size * num_boxes), (-1, 1)), ms.int32)

    def construct(self, loc_data, loc_t, conf_data, conf_t, landm_data, landm_t):
        """Forward pass."""

        # landm loss
        mask_pos1 = ops.cast(self.less(0.0, ops.cast(conf_t, ms.float32)), ms.float32)
        reducesumed = self.expand_dims(self.reduce_sum(mask_pos1), 0)
        n1 = self.maximum(reducesumed, 1)
        mask_pos_idx1 = self.tile(self.expand_dims(mask_pos1, -1), (1, 1, 10))
        loss_landm = self.reduce_sum(self.smooth_l1_loss(landm_data, landm_t) * mask_pos_idx1)
        loss_landm = loss_landm / n1

        # Localization Loss
        mask_pos = ops.cast(self.notequal(0, conf_t), ms.float32)
        conf_t = ops.cast(mask_pos, ms.int32)

        n = self.maximum(self.expand_dims(self.reduce_sum(mask_pos), 0), 1)
        mask_pos_idx = self.tile(self.expand_dims(mask_pos, -1), (1, 1, 4))
        loss_l = self.reduce_sum(self.smooth_l1_loss(loc_data, loc_t) * mask_pos_idx)
        loss_l = loss_l / n

        # Conf Loss
        conf_t_shape = ops.shape(conf_t)
        conf_t = ops.reshape(conf_t, (-1,))
        indices = self.concat((self.idx, ops.reshape(conf_t, (-1, 1))))

        batch_conf = ops.reshape(conf_data, (-1, self.num_classes))
        x_max = self.max(batch_conf)
        loss_c = self.log(self.reduce_sum2(self.exp(batch_conf - x_max), 1)) + x_max
        loss_c = loss_c - ops.reshape(self.gather(batch_conf, indices), (-1, 1))
        loss_c = ops.reshape(loss_c, conf_t_shape)

        # hard example mining
        num_matched_boxes = ops.reshape(self.reduce_sum(mask_pos, 1), (-1,))
        neg_masked_cross_entropy = ops.cast(loss_c * (1 - mask_pos), ms.float32)

        _, loss_idx = self.sort_descend(neg_masked_cross_entropy, self.num_boxes)
        _, relative_position = self.sort(ops.cast(loss_idx, ms.float32), self.num_boxes)
        relative_position = ops.cast(relative_position, ms.float32)
        relative_position = relative_position[:, ::-1]
        relative_position = ops.cast(relative_position, ms.int32)

        num_neg_boxes = self.minimum(num_matched_boxes * self.neg_pre_positive, self.num_boxes - 1)
        tile_num_neg_boxes = self.tile(self.expand_dims(num_neg_boxes, -1), (1, self.num_boxes))
        top_k_neg_mask = ops.cast(self.less(relative_position, tile_num_neg_boxes), ms.float32)

        cross_entropy = self.cross_entropy(batch_conf, conf_t)
        cross_entropy = ops.reshape(cross_entropy, conf_t_shape)

        loss_c = self.reduce_sum(cross_entropy * self.minimum(mask_pos + top_k_neg_mask, 1))

        loss_c = loss_c / n

        return loss_l, loss_c, loss_landm
