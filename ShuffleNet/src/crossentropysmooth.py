# Copyright 2020 Huawei Technologies Co., Ltd
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
"""define loss function for network"""
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.nn.cell import Cell
from mindspore.ops import functional as F
from mindspore.ops import operations as P

class MyLoss(Cell):
    """
    Base class for other losses.
    """
    def __init__(self, reduction='mean'):       # 和pytorch一样，none就返回一个向量，是每个实例的损失
                                                # sum是把损失加起来，mean是加起来再求平均
        super(MyLoss, self).__init__()
        if reduction is None:
            reduction = 'none'

        if reduction not in ('mean', 'sum', 'none'):
            raise ValueError(f"reduction method for {reduction.lower()} is not supported")

        self.average = True
        self.reduce = True
        if reduction == 'sum':
            self.average = False
        if reduction == 'none':
            self.reduce = False

        self.reduce_mean = P.ReduceMean()   # 求平均，并降维
        self.reduce_sum = P.ReduceSum()     # 求和，并降维
        self.mul = P.Mul()                  # 两个tensor逐元素相乘
        self.cast = P.Cast()                # 转换数据类型

    def get_axis(self, x):
        shape = F.shape(x)
        length = F.tuple_len(shape)
        perm = F.make_range(0, length)
        return perm

    def get_loss(self, x, weights=1.0):     # 加权求损失，x应该是包含每个实例的损失的向量
        """
        Computes the weighted loss
        Args:
            weights: Optional `Tensor` whose rank is either 0, or the same rank as inputs, and must be broadcastable to
                inputs (i.e., all dimensions must be either `1`, or the same as the corresponding inputs dimension).
        """
        input_dtype = x.dtype
        x = self.cast(x, mstype.float32)
        weights = self.cast(weights, mstype.float32)
        x = self.mul(weights, x)            
        if self.reduce and self.average:
            x = self.reduce_mean(x, self.get_axis(x))   # 损失求平均
        if self.reduce and not self.average:
            x = self.reduce_sum(x, self.get_axis(x))    # 损失求和
        x = self.cast(x, input_dtype)
        return x

    def construct(self, base, target):
        raise NotImplementedError

class CrossEntropySmooth(MyLoss):
    """CrossEntropy"""
    def __init__(self, sparse=True, reduction='mean', smooth_factor=0., num_classes=1000):
        # sparse指是否用稀疏格式（?）
        super(CrossEntropySmooth, self).__init__()
        self.onehot = P.OneHot()            # 返回one-hot类型的tensor
        self.sparse = sparse
        self.on_value = Tensor(1.0 - smooth_factor, mstype.float32) # smooth_factor和pytroch一样，有一定正则化作用
        self.off_value = Tensor(1.0 * smooth_factor / (num_classes - 1), mstype.float32)
        self.ce = nn.SoftmaxCrossEntropyWithLogits(reduction=reduction)

    def construct(self, logit, label):
        if self.sparse:
            label = self.onehot(label, F.shape(logit)[1], self.on_value, self.off_value)
            # sparse为false，标签就是一个数，将其想来那个华：下标为y处为1，非y处为1/(c-1).
        loss = self.ce(logit, label)    
        return loss
