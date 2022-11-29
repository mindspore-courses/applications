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
"""segmentation"""
import numpy as np
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.tensor import Tensor
import mindspore as ms


class DETRsegm(nn.Cell):
    """
    DETR for segmentation.

    Args:
        detr (Cell): Cell of detr.
        freeze_detr (bool, optional): Freezing parameters if true. Defaults: False.

    Returns:
        Cell, cell instance of DETR segmentation neural network.

    Inputs:
        - **img** (Tensor) - Image after augmentation with shape [batch_size, 3, H, W].
        - **mask** (Tensor) - Image masks after padding with shape [batch_size, H, W].

    Outputs:
        Dict of 3 tensors, the pred_logits and the pred_boxes.

        - **pred_logits** (Tensor) - The classification logits (including no-object) for all queries.
            Shape=[batch_size, num_queries, (num_classes + 1)].
        - **pred_boxes** (Tensor) - The normalized boxes coordinates for all queries, represented as
            (center_x, center_y, height, width). These values are normalized in [0, 1]
        - **pred_masks** (Tensor) - The masks for all queries, Shape=[batch_size, num_queries, H, W].

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.ones((2, 3, 1066, 1201)).astype(np.float32))
        >>> mask = Tensor(np.zeros((2, 1066, 1201)).astype(np.float32))
        >>> net = bulid_detr(num_classes=250, return_interm_layers=True)
        >>> model = DETRsegm(net, freeze_detr=True)
        >>> out = model(x, mask)
    """

    def __init__(self, detr, freeze_detr=False):

        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.detr.get_parameters():
                p.requires_grad = False

        hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.1)
        self.mask_head = MaskHeadSmallConv(hidden_dim + nheads, [1024, 512, 256], hidden_dim)

    def construct(self, img, mask):
        """ Apply DETRsegm """
        bs = img.shape[0]
        features, masks, pos = self.detr.backbone(img, mask)
        src_proj = self.detr.input_proj(features[-1])
        for value in self.detr.query_embed.parameters_dict().values():
            query_embed = Tensor(value)
        hs, memory = self.detr.transformer(src_proj, masks[-1], query_embed, pos[-1])
        outputs_class = self.detr.class_embed(hs)
        outputs_coord = ops.Sigmoid()(self.detr.bbox_embed(hs))
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        bbox_mask = self.bbox_attention(hs[-1], memory, mask=masks[-1])
        seg_masks = self.mask_head(src_proj, bbox_mask, [features[2], features[1], features[0]])
        outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries,
                                           seg_masks.shape[-2], seg_masks.shape[-1])
        out["pred_masks"] = outputs_seg_masks

        return out


def _expand(tensor, length):
    """
    Expand the tensor with the given length.

    Args:
        tensor (Tensor): _description_
        length (int): _description_

    Returns:
        Tensor: Tensor after expanding.
    """
    expand_dims = ops.ExpandDims()
    out = expand_dims(tensor, 1)
    out = ms.numpy.tile(out, (1, int(length), 1, 1, 1))
    out = out.view(tuple([-1] + list(out.shape[2:])))
    return out


class MHAttentionMap(nn.Cell):
    """
    This is a 2D attention module, which only returns the attention softmax (no multiplication by value)

    Args:
        query_dim (int, optional): The channels of the query. Defaults: 256.
        hidden_dim (int, optional): The hidden channels of Dense layer. Defaults: 256.
        num_heads (int, optional): The number of attention head. Defaults: 8.
        dropout (float, optional): The dropout rate. Defaults: 0.1.
        bias (bool, optional): Specifies whether the layer uses a bias vector. Defaults: True.

    Returns:
        Cell, cell instance of MHAttentionMap neural network.

    Inputs:
        - **query** (Tensor) - The first transformer output with shape [batch_size, 100, 256].
        - **key** (Tensor) - The second transformer output with shape [batch_size, 256, H, W].
        - **mask** (Tensor) - Image masks after backbone with shape [batch_size, H, W].

    Outputs:
        weight (Tensor), Tensor with shape [batch_size, 100, num_heads, H, W].

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> query = Tensor(np.ones((2, 100, 256)).astype(np.float32))
        >>> key = Tensor(np.zeros((2, 256, 1066, 1201)).astype(np.float32))
        >>> mask = Tensor(np.zeros((2, 1066, 1201)).astype(np.float32))
        >>> net = MHAttentionMap(256, 256, 8, dropout=0.1)
        >>> out = net(query, key, mask)
    """

    def __init__(self, query_dim=256, hidden_dim=256, num_heads=8, dropout=0.1, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Dense(query_dim, hidden_dim, has_bias=bias)
        self.k_linear = nn.Dense(query_dim, hidden_dim, has_bias=bias)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def construct(self, query, key, mask=None):
        """ Apply MHAttentionMap """
        query = self.q_linear(query)
        conv2d = ops.Conv2D(out_channel=256, kernel_size=1)
        expand_dims = ops.ExpandDims()
        k_weight = expand_dims(self.k_linear.weight, -1)
        k_weight = expand_dims(k_weight, -1)
        k_bias = expand_dims(expand_dims(expand_dims(self.k_linear.bias, -1), -1), 0)
        key = conv2d(key, k_weight) + k_bias
        qh = query.view(query.shape[0], query.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = key.view(key.shape[0], self.num_heads, self.hidden_dim // self.num_heads, key.shape[-2], key.shape[-1])
        qh_np = (qh * self.normalize_fact).asnumpy()
        kh_np = kh.asnumpy()
        weights = np.einsum("bqnc,bnchw->bqnhw", qh_np, kh_np)
        weights = Tensor(weights)
        if mask is not None:
            mask = expand_dims(mask, 1)
            mask = expand_dims(mask, 1)
            _, q_w, n_w, _, _ = weights.shape
            mask = ms.numpy.tile(mask, (1, q_w, n_w, 1, 1))
            weights = ops.MaskedFill()(weights, mask == 1, Tensor(-np.inf * 1.0))
        weights_flat = weights.view((weights.shape[0], weights.shape[1], -1))
        weights = ops.Softmax(axis=-1)(weights_flat).view(weights.shape)
        weights = self.dropout(weights)
        return weights


class MaskHeadSmallConv(nn.Cell):
    """
    Simple convolutional head, using group norm. Upsampling is done using a FPN approach.

    Args:
        dim (int): The input channels equal to the sum of hidden_dim and nheads.
        fpn_dims (list): A list of fpn channels.
        context_dim (list): A list of hidden channels.

    Returns:
        Cell, cell instance of MaskHeadSmallConv neural network.

    Inputs:
        - **x** (Tensor) - The backbone output with shape [batch_size, 256, H, W].
        - **bbox_mask** (Tensor) - The bounding box mask with shape [[batch_size, 100, num_heads, H, W]].
        - **fpns** (list) - Intermediate feature map of backbone.

    Outputs:
        x (Tensor), Tensor with shape [batch_size*100, 16, H*8, W*8].

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> x = Tensor(np.ones((2, 100, 8, 64, 48)).astype(np.float32))
        >>> bbox_mask = Tensor(np.zeros((200, 8, 64, 48)).astype(np.float32))
        >>> mask_head = MaskHeadSmallConv(256 + 8, [1024, 512, 256], 256)
        >>> out = mask_head(x, bbox_mask, [features[2], features[1], features[0]])
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = nn.Conv2d(dim, dim, 3, pad_mode='pad', padding=1, has_bias=True)
        self.gn1 = nn.GroupNorm(8, dim)
        self.lay2 = nn.Conv2d(dim, inter_dims[1], 3, pad_mode='pad', padding=1, has_bias=True)
        self.gn2 = nn.GroupNorm(8, inter_dims[1])
        self.lay3 = nn.Conv2d(inter_dims[1], inter_dims[2], 3, pad_mode='pad', padding=1, has_bias=True)
        self.gn3 = nn.GroupNorm(8, inter_dims[2])
        self.lay4 = nn.Conv2d(inter_dims[2], inter_dims[3], 3, pad_mode='pad', padding=1, has_bias=True)
        self.gn4 = nn.GroupNorm(8, inter_dims[3])
        self.lay5 = nn.Conv2d(inter_dims[3], inter_dims[4], 3, pad_mode='pad', padding=1, has_bias=True)
        self.gn5 = nn.GroupNorm(8, inter_dims[4])
        self.out_lay = nn.Conv2d(inter_dims[4], 1, 3, pad_mode='pad', padding=1, has_bias=True)
        self.dim = dim
        self.adapter1 = nn.Conv2d(fpn_dims[0], inter_dims[1], 1, has_bias=True)
        self.adapter2 = nn.Conv2d(fpn_dims[1], inter_dims[2], 1, has_bias=True)
        self.adapter3 = nn.Conv2d(fpn_dims[2], inter_dims[3], 1, has_bias=True)

    def construct(self, x, bbox_mask, fpns):
        """ Apply MaskHeadSmallConv. """
        bbox_mask_v = bbox_mask.view(tuple([-1] + list(bbox_mask.shape[2:])))
        concat_op = ops.Concat(1)
        x = concat_op((_expand(x, bbox_mask.shape[1]), bbox_mask_v))
        x = self.lay1(x)
        x = self.gn1(x)
        x = ops.ReLU()(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = ops.ReLU()(x)
        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        resize = ops.ResizeNearestNeighbor(cur_fpn.shape[-2:])
        x = cur_fpn + resize(x)
        x = self.lay3(x)
        x = self.gn3(x)
        x = ops.ReLU()(x)
        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        resize = ops.ResizeNearestNeighbor(cur_fpn.shape[-2:])
        x = cur_fpn + resize(x)
        x = self.lay4(x)
        x = self.gn4(x)
        x = ops.ReLU()(x)
        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.shape[0] != x.shape[0]:
            cur_fpn = _expand(cur_fpn, x.shape[0] // cur_fpn.shape[0])
        resize = ops.ResizeNearestNeighbor(cur_fpn.shape[-2:])
        x = cur_fpn + resize(x)
        x = self.lay5(x)
        x = self.gn5(x)
        x = ops.ReLU()(x)
        x = self.out_lay(x)
        return x
