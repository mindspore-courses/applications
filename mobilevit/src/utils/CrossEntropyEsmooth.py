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
"""Softmax Cross Entropy with Logits with label smoothing."""

from mindspore import nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.loss import LossBase
from mindspore.ops import functional as F
from mindspore import ops

from utils.class_factory import ClassFactory, ModuleType


@ClassFactory.register(ModuleType.LOSS)
class CrossEntropySmooth(LossBase):
    """
    Computes softmax cross entropy between digits and labels with label smoothing.

    Measures the distribution error between the probabilities of the input (computed with softmax function) and the
    target where the classes are mutually exclusive using cross entropy loss.

    In order to avoid nan loss in training, smoothing is applied to label.

    Typical input into this function is unnormalized scores denoted as x whose shape is (N, C),
    and the corresponding targets.

    Args:
        classes_num: Number of classes.
        sparse (bool): Specifies whether labels use sparse format or not. Default: False.
        reduction (str): Type of reduction to be applied to loss. The optional values are "mean", "sum", and "none".
            If "none", do not perform reduction. Default: "none".
        smooth_factor: Smoothing the label to avoid nan loss. Default: 0.0.

    Inputs:
        - **logits** (Tensor) - Tensor of shape (N, C). Data type must be float16 or float32.
        - **labels** (Tensor) - Tensor of shape (N, ). If `sparse` is True, The type of
          `labels` is int32 or int64. If `sparse` is False, the type of `labels` is the same as the type of `logits`.

    Outputs:
        Tensor, a tensor of the same shape and type as logits with the component-wise logistic losses.
    """

    def __init__(self, classes_num, sparse=True, reduction='mean', smooth_factor=0.):
        super(CrossEntropySmooth, self).__init__()
        self.onehot = ops.OneHot()
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32)
        self.off_value = Tensor(1.0 * smooth_factor / (classes_num - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
        loss = self.ce(logit, label)
        return loss
