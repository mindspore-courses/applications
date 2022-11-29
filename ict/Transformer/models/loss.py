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
"""
Define the model with loss.
"""

import mindspore.nn as nn
import mindspore.ops.operations as P


class TransformerWithLoss(nn.Cell):
    """
    Wrap the network with loss function to return Transformer with loss.

    Args:
        backbone (Cell): The target network to wrap.
    """

    def __init__(self, backbone):
        super(TransformerWithLoss, self).__init__(auto_prefix=False)
        self.backbone = backbone
        self.loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True)

    def construct(self, x, targets, masks):
        logits = self.backbone(x, masks)
        loss = self.loss_fn(logits.view(-1, logits.shape[-1]), targets.view(-1))
        masks = P.ExpandDims()(masks, 2)
        masks = masks.view(-1)
        loss *= masks
        loss = P.ReduceMean()(loss)
        return loss
