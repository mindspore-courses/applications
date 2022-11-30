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
"""TransformerEncoder module."""

import numpy as np

from mindspore import nn

from models.blocks.drop_path import DropPath
from models.blocks import Attention, FeedForward, ResidualCell
__all__ = ['TransformerEncoder']


class TransformerEncoder(nn.Cell):
    """
    Transformer Encoder module with multi-layer stacked of `TransformerEncoderLayer`, including multihead self
    attention and feedforward layer.

    Args:
        dim (int): The dimension of embedding.
        num_layers (int): The depth of transformer.
        num_heads (int): The number of attention heads.
        mlp_dim (int): The dimension of MLP hidden layer.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
        attention_keep_prob (float): The keep rate for attention. Default: 1.0.
        drop_path_keep_prob (float): The keep rate for drop path. Default: 1.0.
        activation (nn.Cell): Activation function which will be stacked on top of the
        normalization layer (if not None), otherwise on top of the conv layer. Default: nn.GELU.
        norm (nn.Cell, optional): Norm layer that will be stacked on top of the convolution
        layer. Default: nn.LayerNorm.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = TransformerEncoder(768, 12, 12, 3072)
    """

    def __init__(self,
                 dim: int,
                 num_layers: int,
                 num_heads: int,
                 mlp_dim: int,
                 keep_prob: float = 1.,
                 attention_keep_prob: float = 1.0,
                 drop_path_keep_prob: float = 1.0,
                 activation: nn.Cell = nn.GELU,
                 norm: nn.Cell = nn.LayerNorm):
        super(TransformerEncoder, self).__init__()
        drop_path_rate = 1 - drop_path_keep_prob
        dpr = [i.item() for i in np.linspace(0, drop_path_rate, num_layers)]
        attn_seeds = [np.random.randint(1024) for _ in range(num_layers)]
        mlp_seeds = [np.random.randint(1024) for _ in range(num_layers)]

        layers = []
        for i in range(num_layers):
            normalization1 = norm((dim,))
            normalization2 = norm((dim,))
            attention = Attention(dim=dim,
                                  num_heads=num_heads,
                                  keep_prob=keep_prob,
                                  attention_keep_prob=attention_keep_prob)

            feedforward = FeedForward(in_features=dim,
                                      hidden_features=mlp_dim,
                                      activation=activation,
                                      keep_prob=keep_prob)

            if drop_path_rate > 0:
                layers.append(
                    nn.SequentialCell([
                        ResidualCell(nn.SequentialCell([normalization1,
                                                        attention,
                                                        DropPath(dpr[i], attn_seeds[i])])),
                        ResidualCell(nn.SequentialCell([normalization2,
                                                        feedforward,
                                                        DropPath(dpr[i], mlp_seeds[i])]))]))
            else:
                layers.append(
                    nn.SequentialCell([
                        ResidualCell(nn.SequentialCell([normalization1,
                                                        attention])),
                        ResidualCell(nn.SequentialCell([normalization2,
                                                        feedforward]))
                    ])
                )
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        """Transformer construct."""
        return self.layers(x)
