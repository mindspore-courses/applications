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
"""Transformer network"""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.common.initializer import Normal


class CausalSelfAttention(nn.Cell):
    """
    The CausalSelfAttention part of transformer.

    Args:
        n_embd (int): The size of the vector space in which words are embedded.
        n_head (int): The number of multi-head.
        block_size (int): The context size(Input sequence length).
        resid_pdrop (float): The probability of resid_pdrop. Default: 0.1
        attn_pdrop (float): The probability of attn_pdrop. Default: 0.1

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, resid_pdrop: float = 0.1, attn_pdrop: float = 0.1):
        super().__init__()
        # key, query, value projections for all heads
        self.key = nn.Dense(in_channels=n_embd, out_channels=n_embd,
                            weight_init=Normal(sigma=0.02, mean=0.0))
        self.query = nn.Dense(in_channels=n_embd, out_channels=n_embd,
                              weight_init=Normal(sigma=0.02, mean=0.0))
        self.value = nn.Dense(in_channels=n_embd, out_channels=n_embd,
                              weight_init=Normal(sigma=0.02, mean=0.0))
        # regularization
        self.attn_drop = nn.Dropout(keep_prob=1.0 - attn_pdrop)
        self.resid_drop = nn.Dropout(keep_prob=1.0 - resid_pdrop)
        # output projection
        self.proj = nn.Dense(in_channels=n_embd, out_channels=n_embd,
                             weight_init=Normal(sigma=0.02, mean=0.0))

        tril = nn.Tril()
        self.mask = mindspore.Parameter(
            tril(P.Ones()((block_size, block_size), mindspore.float32)).view(1, 1, block_size, block_size),
            requires_grad=False)

        self.n_head = n_head

    def construct(self, x):
        """SelfAttention forward."""
        b, t, c = P.Shape()(x)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(b, t, self.n_head, c // self.n_head)
        k = k.transpose(0, 2, 1, 3)
        q = self.query(x).view(b, t, self.n_head, c // self.n_head)
        q = q.transpose(0, 2, 1, 3)
        v = self.value(x).view(b, t, self.n_head, c // self.n_head)
        v = v.transpose(0, 2, 1, 3)

        # causal self-attention; Self-attend: (b, nh, t, hs) x (b, nh, hs, t) -> (b, nh, t, t)
        k_shape = k.shape[-1]
        sz = 1.0 / (Tensor(k_shape) ** Tensor(0.5))
        att = (ops.matmul(q, k.transpose(0, 1, 3, 2)) * sz)
        att = P.Softmax()(att)
        att = self.attn_drop(att)
        y = mindspore.ops.matmul(att, v)
        y = y.transpose(0, 2, 1, 3).view(b, t, c)  # re-assemble all head outputs side by side
        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class GELU2(nn.Cell):
    """
    The new gelu2 activation function.

    Returns:
        Tensor, output tensor.
    """

    def construct(self, x):
        return x * P.Sigmoid()(1.702 * x)


class Block2(nn.Cell):
    """
    Transformer block with original GELU2.

    Args:
        n_embd (int): The size of the vector space in which words are embedded.
        n_head (int): The number of multi-head.
        block_size (int): The context size(Input sequence length).
        resid_pdrop (float): The probability of resid_pdrop. Default: 0.1
        attn_pdrop (float): The probability of attn_pdrop. Default: 0.1

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, resid_pdrop: float = 0.1, attn_pdrop: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=[n_embd], epsilon=1e-05)
        self.ln2 = nn.LayerNorm(normalized_shape=[n_embd], epsilon=1e-05)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, resid_pdrop, attn_pdrop)
        self.mlp = nn.SequentialCell([
            nn.Dense(in_channels=n_embd, out_channels=4 * n_embd,
                     weight_init=Normal(sigma=0.02, mean=0.0)),
            GELU2(),
            nn.Dense(in_channels=4 * n_embd, out_channels=n_embd,
                     weight_init=Normal(sigma=0.02, mean=0.0)),
            nn.Dropout(keep_prob=1.0 - resid_pdrop),
        ])

    def construct(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class Block(nn.Cell):
    """
    Transformer block with original GELU.

    Args:
        n_embd (int): The size of the vector space in which words are embedded.
        n_head (int): The number of multi-head.
        block_size (int): The context size(Input sequence length).
        resid_pdrop (float): The probability of resid_pdrop. Default: 0.1
        attn_pdrop (float): The probability of attn_pdrop. Default: 0.1

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, n_embd: int, n_head: int, block_size: int, resid_pdrop: float = 0.1, attn_pdrop: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(normalized_shape=[n_embd], epsilon=1e-05)
        self.ln2 = nn.LayerNorm(normalized_shape=[n_embd], epsilon=1e-05)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, resid_pdrop, attn_pdrop)
        self.mlp = nn.SequentialCell([
            nn.Dense(in_channels=n_embd, out_channels=4 * n_embd,
                     weight_init=Normal(sigma=0.02, mean=0.0)),
            nn.GELU(),
            nn.Dense(in_channels=4 * n_embd, out_channels=n_embd,
                     weight_init=Normal(sigma=0.02, mean=0.0)),
            nn.Dropout(keep_prob=1.0 - resid_pdrop),
        ])

    def construct(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Cell):
    """
    The full GPT language model, with a context size of block_size.

    Args:
        vocab_size (int): The size of the vocabulary in the embedded data.
        n_embd (int): The size of the vector space in which words are embedded.
        n_layer (int): The number of attention layer.
        n_head (int): The number of multi-head.
        block_size (int): The context size(Input sequence length).
        use_gelu2 (bool): Use the new gelu2 activation function.
        embd_pdrop (float): The probability of embd_pdrop. Default: 0.1
        resid_pdrop (float): The probability of resid_pdrop. Default: 0.1
        attn_pdrop (float): The probability of attn_pdrop. Default: 0.1

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, vocab_size: int, n_embd: int, n_layer: int, n_head: int, block_size: int, use_gelu2: bool,
                 embd_pdrop: float = 0.1, resid_pdrop: float = 0.1, attn_pdrop: float = 0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd, embedding_table=Normal(sigma=0.02, mean=0.0))
        self.pos_emb = mindspore.Parameter(P.Zeros()((1, block_size, n_embd), mindspore.float32))
        self.drop = nn.Dropout(keep_prob=1.0 - embd_pdrop)
        # transformer
        if use_gelu2:
            self.blocks = nn.SequentialCell(
                [*[Block2(n_embd, n_head, block_size, resid_pdrop, attn_pdrop) for _ in range(n_layer)]])
        else:
            self.blocks = nn.SequentialCell(
                [*[Block(n_embd, n_head, block_size, resid_pdrop, attn_pdrop) for _ in range(n_layer)]])
        # decoder head
        self.ln_f = nn.LayerNorm(normalized_shape=[n_embd], epsilon=1e-05)
        self.head = nn.Dense(in_channels=n_embd, out_channels=vocab_size, has_bias=False,
                             weight_init=Normal(sigma=0.02, mean=0.0))

        self.block_size = block_size

    def get_block_size(self):
        return self.block_size

    def construct(self, idx, masks):
        """Transformer forward and get the image pixel probability."""
        _, t = idx.shape
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        masks = P.ExpandDims()(masks, 2)
        token_embeddings = token_embeddings * (1 - masks)
        position_embeddings = self.pos_emb[:, :t, :]  # each position maps to a (learnable) vector
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits
