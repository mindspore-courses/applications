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
"""Build detr model"""
import mindspore.nn as nn
from mindspore import ops
from mindspore.common.tensor import Tensor

from .backbone import build_backbone
from .transformer import build_transformer


class DETR(nn.Cell):
    """
    This is the DETR module that performs object detection.

    Args:
        backbone (Cell): Resnet-based backbone.
        transformer (Cell): Cell of transformer.
        num_classes (int, optional): Number of object classes. Defaults: 91.
        num_queries (int, optional): Number of object queries. This is the maximal number of objects
                                     DETR can detect in a single image. Defaults: 100.

    Returns:
        Cell, cell instance of DETR neural network.

    Inputs:
        - **src** (Tensor) - Image after augmentation with shape [batch_size, 3, H, W].
        - **mask** (Tensor) - Image masks after padding with shape [batch_size, H, W].

    Outputs:
        Dict of 2 tensors, the pred_logits and the pred_boxes.

        - **pred_logits** (Tensor) - The classification logits (including no-object) for all queries.
            Shape= [batch_size, num_queries, (num_classes + 1)].
        - **pred_boxes** (Tensor) - The normalized boxes coordinates for all queries, represented as
            (center_x, center_y, height, width). These values are normalized in [0, 1]

    Supported Platforms:
        ``GPU`` ``CPU``

    Examples:
        >>> net = DETR(backbone, tf, num_classes=91, num_queries=100)
        >>> x = Tensor(np.ones((2, 3, 800, 994)).astype(np.float32))
        >>> mask = Tensor(np.zeros((2, 800, 994)).astype(np.float32))
        >>> out = net(x, mask)
    """

    def __init__(self, backbone, transformer, num_classes=91, num_queries=100):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.class_embed = nn.Dense(hidden_dim, num_classes + 1)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1,
                                    has_bias=True, pad_mode='valid')
        self.backbone = backbone

    def construct(self, src, mask):
        """ Apply DETR """
        features, mask, pos = self.backbone(src, mask)
        feature = self.input_proj(features[-1])
        query_embed = Tensor(self.query_embed.trainable_params()[0])
        hs = self.transformer(feature, mask[-1], query_embed, pos[-1])[0]
        outputs_class = self.class_embed(hs)
        sigmoid = ops.Sigmoid()
        outputs_coord = sigmoid(self.bbox_embed(hs))
        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        return out


class MLP(nn.Cell):
    """
    Very simple multi-layer perceptron (also called FFN)

    Args:
        input_dim (int): The channel number of the input tensor.
        hidden_dim (int): The hidden size of the Dense layer.
        output_dim (int): The channel number of the output tensor.
        num_layers (int): The number of layers.

    Inputs:
        - **x** (Tensor) - Tensor of input with channel input_dim.

    Outputs:
        Tensor of output with channel output_dim.

    Examples:
        >>> net = MLP(256, 256, 4, 3)
        >>> x = Tensor(np.ones((6, 2, 100, 256)).astype(np.float32))
        >>> out = net(x)
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.CellList([nn.Dense(n, k) for n, k in zip([input_dim] + h, h + [output_dim])])

    def construct(self, x):
        """Apply MLP"""
        for i, layer in enumerate(self.layers):
            relu = ops.ReLU()
            x = relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def bulid_detr(resnet='resnet50', return_interm_layers=False, num_classes=91, is_dilation=False):
    """
    Get detr.

    Args:
        resnet (str, optional): Type of resnet. Defaults: 'resnet50'.
        return_interm_layers (bool, optional): Get intern layer outputs of resnet if True. Defaults: False.
        num_classes (int, optional): Number of object classes. Defaults: 91.
        is_dilation (bool, optional): Use dilated Convolution. Defaults: False.

    Returns:
        Cell: detr neural network.

    Examples:
        >>> net = bulid_detr(resnet='resnet50', return_interm_layers=False, num_classes=91, is_dilation=False)
    """
    backbone = build_backbone(resnet=resnet, return_interm_layers=return_interm_layers,
                              is_dilation=is_dilation)
    tf = build_transformer()
    net = DETR(backbone, tf, num_classes=num_classes, num_queries=100)
    return net
