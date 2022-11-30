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
"""Feed Forward module."""

from typing import Optional

from mindspore import nn


class FeedForward(nn.Cell):
    """
    Feed Forward layer implementation.

    Args:
        in_features (int): The dimension of input features.
        hidden_features (int): The dimension of hidden features. Default: None.
        out_features (int): The dimension of output features. Default: None
        activation (nn.Cell): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = FeedForward(768, 3072)
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None,
                 activation: nn.Cell = nn.GELU,
                 keep_prob: float = 1.0):
        super(FeedForward, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dense1 = nn.Dense(in_features, hidden_features)
        self.activation = activation()
        self.dense2 = nn.Dense(hidden_features, out_features)
        self.dropout = nn.Dropout(keep_prob)

    def construct(self, x):
        """Feed Forward construct."""
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.dropout(x)

        return x
