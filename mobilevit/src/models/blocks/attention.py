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
"""Attention module."""

from mindspore import nn, Tensor
from mindspore import ops
from mindspore._checkparam import Validator


class Attention(nn.Cell):
    """
    Attention layer implementation, Rearrange Input -> B x N x hidden size.

    Args:
        dim (int): The dimension of input features.
        num_heads (int): The number of attention heads. Default: 8.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.
        attention_keep_prob (float): The keep rate for attention. Default: 1.0.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = Attention(768, 12)
    """

    def __init__(self,
                 dim: int,
                 num_heads: int = 8,
                 keep_prob: float = 1.0,
                 attention_keep_prob: float = 1.0):
        super(Attention, self).__init__()
        Validator.check_equal_int(dim % num_heads, 0, 'dim should be divisible by num_heads.')
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = Tensor(head_dim ** -0.5)

        self.qkv = nn.Dense(dim, dim * 3)
        self.attn_drop = nn.Dropout(attention_keep_prob)
        self.out = nn.Dense(dim, dim)
        self.out_drop = nn.Dropout(keep_prob)

        self.mul = ops.Mul()
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.unstack = ops.Unstack(axis=0)
        self.attn_matmul_v = ops.BatchMatMul()
        self.q_matmul_k = ops.BatchMatMul(transpose_b=True)
        self.softmax = nn.Softmax(axis=-1)

    def construct(self, x):
        """Attention construct."""
        b, n, c = x.shape
        qkv = self.qkv(x)
        qkv = self.reshape(qkv, (b, n, 3, self.num_heads, c // self.num_heads))
        qkv = self.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = self.unstack(qkv)

        attn = self.q_matmul_k(q, k)
        attn = self.mul(attn, self.scale)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        out = self.attn_matmul_v(attn, v)
        out = self.transpose(out, (0, 2, 1, 3))
        out = self.reshape(out, (b, n, c))
        out = self.out(out)
        out = self.out_drop(out)

        return out
