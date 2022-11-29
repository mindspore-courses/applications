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
"""transformer api"""
import copy

from mindspore import ops
import mindspore.nn as nn
import mindspore as ms
from mindspore.common.tensor import Tensor


class MultiHeadAttention(nn.Cell):
    r"""
    This is an implementation of multihead attention in the paper `Attention is all you need
    <https://arxiv.org/pdf/1706.03762v5.pdf>`. Given the query vector with source length, and the
    key and value vector with target length, the attention will be performed as the following

    .. math::
            MultiHeadAttention(query, key, vector) = Concat(head_1, \dots, head_h)W^O

    where :math:`head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)`. The default is with a bias.

    if query, key and value tensor is same, then it will be self attention.
    Args:
        n_head (int): The number of the heads.
        d_model (int): The hidden size of the input.
        dropout (float, optional): The dropout rate of the attention scores. Defaults to 0.1.

    Inputs:
        - **query** (Tensor) - The query vector with shape (src_seq_length, batch_size, hidden_size).
        - **key** (Tensor) - The key vector with shape (tgt_seq_length, batch_size, hidden_size).
        - **value** (Tensor) - The value vector with shape (tgt_seq_length, batch_size, hidden_size).
        - **mask** (Tensor) - The attention mask matrix with shape (batch_size, tgt_seq_length).

    Outputs:
        Tuple, a tuple contains(`output`, `attention`)

        - **output** (Tensor) - Tensor, the float tensor of the output of the layer with
            shape (src_seq_length, batch_size, hidden_size).

        - **attention** (Tensor) - A Tensor with shape (batch_size, num_heads, size_per_head, tgt_seq_length).

    Supported Platforms:
    ``CPU`` ``GPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore.common.tensor import Tensor
        >>> k = v = Tensor(np.ones((20, 2, 256)).astype(np.float32))
        >>> q = Tensor(np.ones((30, 2, 256)).astype(np.float32))
        >>> mha = MultiHeadAttention(n_head=8, d_model=256)
        >>> mask = Tensor(np.ones((2, k.shape[0])).astype(np.float32))
        >>> output, attn = mha(q, k, v, mask)
    """

    def __init__(self, n_head, d_model, dropout=0.1):
        super().__init__()
        self.hid_dim = d_model
        self.n_heads = n_head
        self.w_qs = nn.Dense(d_model, d_model)
        self.w_ks = nn.Dense(d_model, d_model)
        self.w_vs = nn.Dense(d_model, d_model)
        self.out_proj = nn.Dense(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = (d_model // n_head)**0.5

    def construct(self, query, key, value, mask=None):
        """ Apply  MultiHeadAttention"""
        transpose = ops.Transpose()
        query = transpose(query, (1, 0, 2))
        key = transpose(key, (1, 0, 2))
        value = transpose(value, (1, 0, 2))

        bs = query.shape[0]
        q_w = self.w_qs(query)
        k_w = self.w_ks(key)
        v_w = self.w_vs(value)

        q_w = transpose(q_w.view(bs, -1, self.n_heads, self.hid_dim // self.n_heads), (0, 2, 1, 3))
        k_w = transpose(k_w.view(bs, -1, self.n_heads, self.hid_dim // self.n_heads), (0, 2, 1, 3))
        v_w = transpose(v_w.view(bs, -1, self.n_heads, self.hid_dim // self.n_heads), (0, 2, 1, 3))

        batmatmul = ops.BatchMatMul()
        attention = batmatmul(q_w, transpose(k_w, (0, 1, 3, 2))) / self.scale

        if mask is not None:
            broadcast_to = ops.BroadcastTo(attention.shape)
            mask = broadcast_to(mask[:, None, None, :])
            attention = ops.MaskedFill()(attention, mask == 1, Tensor(-1e-20))

        softmax = ops.Softmax(-1)
        attention = self.dropout(softmax(attention))
        output1 = batmatmul(attention, v_w)
        output2 = transpose(output1, (0, 2, 1, 3))
        output3 = output2.view(bs, -1, self.n_heads * (self.hid_dim // self.n_heads))
        output4 = self.out_proj(output3)
        output = transpose(output4, (1, 0, 2))
        return output, attention


def _get_activation_fn(activation):
    """
    Return an activation function given a string.

    Args:
        activation (str): Activation style

    Raises:
        RuntimeError: If activation is not relu or gelu.

    Returns:
        Cell: An activation function
    """
    if activation == "relu":
        return ops.ReLU()
    if activation == "gelu":
        return ops.GeLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, num_layers):
    """
    Clone layers

    Args:
        module (Cell): Encoderlayer of decoderlayer
        num_layers (int): Number of layer

    Returns:
        CellList: CellList of layers.
    """
    return nn.CellList([copy.deepcopy(module) for _ in range(num_layers)])


class TransformerEncoderLayer(nn.Cell):
    """
    Transformer Encoder Layer. This is an implementation of the single layer of the transformer
    encoder layer, including multihead attention and feedward layer.

    Args:
        d_model (int, optional): The hidden size of the input. Defaults to 256.
        nhead (int, optional): The number of the heads. Defaults to 8.
        dim_feedforward (int, optional): The hidden size of the feedforward layer. Defaults to 2048.
        dropout (float, optional): The dropout rate of the attention scores. Defaults to 0.1.
        activation (str, optional): The activation of the internal feedforward layer. Defaults to "relu".

    Inputs:
        - **src** (Tensor) - Float Tensor, shape should be (src_seq_length, batch_size, hidden_size).
        - **src_padding_mask** (Tensor) - Attention mask with shape (batch_size, src_seq_length)
        - **pos** (Tensor) - Position embedding with shape (src_seq_length, batch_size, hidden_size).

    Outputs:
        Tensor, the float tensor of the output of the layer with shape (seq_length, batch_size, hidden_size).

    Supported Platforms:
        ``CPU`` ``GPU``

    """

    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(nhead, d_model, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm((d_model,))
        self.norm2 = nn.LayerNorm((d_model,))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def construct(self, src, src_padding_mask=None, pos=None):
        """ Apply TransformerEncoderLayer """
        query = key = src + pos
        src2 = self.self_attn(query, key, src, mask=src_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # FFN
        src2 = self.activation(self.linear1(src))
        src2 = self.linear2(self.dropout(src2))

        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Cell):
    """
    Transformer Encoder module with multi-layer stacked of `TransformerEncoderLayer`, including multihead self
    attention and feedforward layer.

    Args:
        encoder_layer (int): The layers of the TransformerEncoderLayer.
        num_layers (int): The number of encoder layer.
        norm (Cell, optional): The norm for output. Defaults to None.

    Inputs:
        - **src** (Tensor) - Float Tensor, shape should be (src_seq_length, batch_size, hidden_size).
        - **src_padding_mask** (Tensor) - Attention mask with shape (batch_size, src_seq_length)
        - **pos** (Tensor) - Position embedding with shape (src_seq_length, batch_size, hidden_size).

    Outputs:
        Tensor, the float tensor of the output of the layer with shape (seq_length, batch_size, hidden_size).

    Supported Platforms:
        ``CPU`` ``GPU``
    """

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def construct(self, src, src_padding_mask=None, pos=None):
        """ Apply TransformerEncoder """
        output = src
        for layer in self.layers:
            output = layer(output, src_padding_mask=src_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoderLayer(nn.Cell):
    """
    Transformer Decoder Layer. This is an implementation of the single layer of the transformer
    decoder layer, including multihead attention and feedward layer.

    Args:
        d_model (int, optional): The hidden size of the input. Defaults to 256.
        nhead (int, optional): The number of the heads. Defaults to 8.
        dim_feedforward (int, optional): The hidden size of the feedforward layer. Defaults to 2048.
        dropout (float, optional): The dropout rate of the attention scores. Defaults to 0.1.
        activation (str, optional): The activation of the internal feedforward layer. Defaults to "relu".

    Inputs:
        - **target** (Tensor) - Float Tensor, shape should be (100, batch_size, hidden_size).
        - **memory** (Tensor) - Outputs of encoder with shape (src_seq_length, batch_size, hidden_size)
        - **target_padding_mask** (Tensor) - The attention mask for decoder.
        - **memory_padding_mask** (Tensor) - The memory mask with shape (batch_size, src_seq_length).
        - **pos** (Tensor) - Position embedding with shape (src_seq_length, batch_size, hidden_size).
        - **query_pos** (Tensor) - Query position embedding with shape (100, batch_size, hidden_size).

    Outputs:
        Tensor, the float tensor of the output of the layer with shape (100, batch_size, hidden_size).

    Supported Platforms:
        ``CPU`` ``GPU``
    """

    def __init__(self, d_model=256, nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu"):
        super().__init__()
        self.self_attn = MultiHeadAttention(n_head=nhead, d_model=d_model, dropout=dropout)
        self.multihead_attn = MultiHeadAttention(n_head=nhead, d_model=d_model, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Dense(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Dense(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm((d_model,))
        self.norm2 = nn.LayerNorm((d_model,))
        self.norm3 = nn.LayerNorm((d_model,))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def construct(self, target, memory, target_padding_mask=None,
                  memory_padding_mask=None, pos=None, query_pos=None):
        """ Apply TransformerDecoderLayer """
        query = key = target + query_pos
        target2 = self.self_attn(query, key, target, mask=target_padding_mask)[0]

        target = target + self.dropout1(target2)
        target = self.norm1(target)
        query = target + query_pos
        key = memory + pos
        target2 = self.multihead_attn(query, key, memory, mask=memory_padding_mask)[0]
        target = target + self.dropout2(target2)
        target = self.norm2(target)
        # FFN
        target2 = self.activation(self.linear1(target))
        target2 = self.linear2(self.dropout(target2))

        target = target + self.dropout3(target2)
        target = self.norm3(target)
        return target


class TransformerDecoder(nn.Cell):
    """
    Transformer Decoder module with multi-layer stacked of `TransformerDecoderLayer`, including multihead self
    attention and feedforward layer.

    Args:
        decoder_layer (int): The layers of the TransformerDecoderLayer.
        num_layers (int): The number of decoder layer.
        norm (Cell, optional): The norm for output. Defaults to None.
        return_intermediate (bool, optional): Return intermediate result if true. Defaults to False.

    Inputs:
        - **target** (Tensor) - Float Tensor, shape should be (100, batch_size, hidden_size).
        - **memory** (Tensor) - Outputs of encoder with shape (src_seq_length, batch_size, hidden_size)
        - **target_padding_mask** (Tensor) - The attention mask for decoder.
        - **memory_padding_mask** (Tensor) - The memory mask with shape (batch_size, src_seq_length).
        - **pos** (Tensor) - Position embedding with shape (src_seq_length, batch_size, hidden_size).
        - **query_pos** (Tensor) - Query position embedding with shape (100, batch_size, hidden_size).

    Outputs:
        Tensor, the float tensor of the output of the layer with shape (num_layers, 100, batch_size, hidden_size).

    Supported Platforms:
        ``CPU`` ``GPU``
    """

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def construct(self, target, memory, target_padding_mask=None,
                  memory_padding_mask=None, pos=None, query_pos=None):
        """ Apply TransformerDecoder """
        output = target
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, target_padding_mask,
                           memory_padding_mask, pos, query_pos)

            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate[-1] = output

        if self.return_intermediate:
            stack = ops.Stack()
            return stack(intermediate)

        expand_dims = ops.ExpandDims()
        output = expand_dims(output, 0)
        return output


class Transformer(nn.Cell):
    r"""
    Transformer module including encoder and decoder. And the default hidden act is `relu`.
    The details can be found in `Attention is all you need <https://arxiv.org/pdf/1706.03762v5.pdf>`_.

    Args:
        d_model (int, optional): The hidden size of the input. Defaults to 256.
        nhead (int, optional): The number of the heads. Defaults to 8.
        num_encoder_layers (int, optional): The number of encoder layer. Defaults to 6.
        num_decoder_layers (int, optional): The number of decoder layer. Defaults to 6.
        dim_feedforward (int, optional): The hidden size of the feedforward layer. Defaults to 2048.
        dropout (float, optional): The dropout rate of the attention scores. Defaults to 0.1.
        activation (str, optional): The activation of the internal feedforward layer. Defaults to "relu".
        return_intermediate_dec (bool, optional): Return intermediate result if true. Defaults to False.

    Inputs:
        - **src** (Tensor) - Float Tensor, shape should be (src_seq_length, batch_size, hidden_size).
        - **mask** (Tensor) - Outputs of encoder with shape (batch_size, hidden_size)
        - **query_embed** (Tensor) - Query position embedding with shape (100, batch_size, hidden_size).
        - **pos_embed** (Tensor) - Position embedding with shape (src_seq_length, batch_size, hidden_size).

    Outputs:
        - **hs** (Tensor) - Tensor with shape (num_layers, 100, batch_size, hidden_size).
        - **memory** (Tensor) - Tensor with shape (batch_size, hidden_size, h, w).

    Supported Platforms:
        ``CPU`` ``GPU``
    """

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        encoder_norm = None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm((d_model,))
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        self.d_model = d_model
        self.nhead = nhead

    def construct(self, src, mask, query_embed, pos_embed):
        """ Apply Transformer """
        bs, c, h, w = src.shape
        transpose = ops.Transpose()
        src = transpose(src.view(bs, c, -1), (2, 0, 1))
        pos_embed = transpose(pos_embed.view(bs, c, -1), (2, 0, 1))
        expand_dims = ops.ExpandDims()
        query_embed = expand_dims(query_embed, 1)
        query_embed = ms.numpy.tile(query_embed, (1, bs, 1))
        mask = ops.Flatten()(mask)
        zeroslike = ops.ZerosLike()
        tgt = zeroslike(query_embed)
        memory = self.encoder(src, src_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hs = transpose(hs, (0, 2, 1, 3))
        memory = transpose(memory, (1, 2, 0)).view((bs, c, h, w))
        return hs, memory


def build_transformer(hidden_dim=256, dropout=0.1, nheads=8, dim_feedforward=2048,
                      enc_layers=6, dec_layers=6, return_intermediate_dec=True):
    """
    Build transformer

    Args:
        hidden_dim (int, optional): The hidden size of the input. Defaults to 256.
        dropout (float, optional): The dropout rate of the attention scores. Defaults to 0.1.
        nheads (int, optional): The number of the heads. Defaults to 8.
        dim_feedforward (int, optional): The hidden size of the feedforward layer. Defaults to 2048.
        enc_layers (int, optional): The number of encoder layer. Defaults to 6.
        dec_layers (int, optional): The number of decoder layer. Defaults to 6.
        return_intermediate_dec (bool, optional): Return intermediate result if true. Defaults to True.

    Returns:
        Cell: Transformer
    """
    return Transformer(
        d_model=hidden_dim,
        dropout=dropout,
        nhead=nheads,
        dim_feedforward=dim_feedforward,
        num_encoder_layers=enc_layers,
        num_decoder_layers=dec_layers,
        return_intermediate_dec=return_intermediate_dec,
    )
