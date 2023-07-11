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
"""Build backbone """
import mindspore.nn as nn
from mindspore import ops

from .position_encoding import PositionEmbeddingSine
from .resnet import resnet50, resnet101


class Joiner(nn.Cell):
    """
    Joiner architecture.

    Args:
        backbone (Cell): resnet backbone for network.
        position_embedding (Cell): position embedding.
        num_channels (int, optional): Channel of output feature. Defaults to 2048.

    Inputs:
        - **img** (Tensor) - Image after augmentation.
        - **mask** (Tensor) - Image masks after padding.

    Outputs:
        Tuple of 3 tensors, the feature, mask and the pos_emb.

        - **feature** (Tensor) - Feature generated by resnet.
        - **mask** (Tensor) - Mask of feature.
        - **pos_emb** (Tensor) - Tensor of Position embedding.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> backbone = Joiner(resnet, posEmb)
    """

    def __init__(self, backbone, position_embedding, num_channels=2048):
        super(Joiner, self).__init__()
        self.backone = backbone
        self.position_embedding = position_embedding
        self.num_channels = num_channels

    def construct(self, img, mask):
        """Apply Joiner architecture"""
        feature = self.backone(img)
        expand_dims = ops.ExpandDims()
        squeeze = ops.Squeeze(1)
        mk = expand_dims(mask, 1)
        pos_emb = []
        mask = []
        for i in feature:
            resize = ops.ResizeNearestNeighbor(i.shape[-2:])
            mk = resize(mk)
            mask.append(squeeze(mk))
            pos = self.position_embedding(squeeze(mk))
            pos_emb.append(pos)
        return feature, mask, pos_emb


def build_backbone(resnet='resnet50', return_interm_layers=False, is_dilation=False):
    """
    Get backbone neural network.

    Args:
        resnet (str): resnet style.
        return_interm_layers (bool): return interm_layers result.
        is_dilation (bool): use dilation conv in the last layer.

    Returns:
        Cell, cell instance of backbone neural network.

    Examples:
        >>> net = build_backbone(resnet='resnet50')
    """
    if resnet == 'resnet50':
        resnet = resnet50(return_interm_layers=return_interm_layers, is_dilation=is_dilation)
    elif resnet == 'resnet101':
        resnet = resnet101(return_interm_layers=return_interm_layers, is_dilation=is_dilation)
    else:
        raise RuntimeError(f"resnet should be resnet50/resnet101, not {resnet}.")
    posemb = PositionEmbeddingSine()
    backbone = Joiner(resnet, posemb)
    return backbone