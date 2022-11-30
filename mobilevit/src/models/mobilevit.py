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
"""MobileViT backbone"""
from collections import OrderedDict
from typing import Dict, Tuple, Optional
from mindspore import nn, Tensor

from models.blocks.convnormactivation import ConvNormActivation
from models.blocks.globalavgpooling import GlobalAvgPooling
from models.swish import Swish
from models.invertedresidual import InvertedResidual
from models.mobilevit_block import MobileViTBlock
from config.mobilevit_config import get_configuration


class MobileViT(nn.Cell):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`
    Args:
        num_classes(int):The number of classification. Default: 1000.
        classifier_dropout(float): The drop out rate. Default: 0.1.
        image_channels(int):Input channel. Default: 3.
        out_channels(int):Out channel. Default: 16.
        model_type(str):specifications of the model.Default: xx_small
    """

    def __init__(self,
                 num_classes: int = 1000,
                 classifier_dropout: float = 0.1,
                 image_channels: int = 3,
                 out_channels: int = 16,
                 model_type: str = "xx_small",
                 ):
        self.mobilevit_config = get_configuration(model_type)
        super(MobileViT, self).__init__()

        self.conv_1 = ConvNormActivation(
            in_planes=image_channels,
            out_planes=out_channels,
            kernel_size=3,
            stride=2,
            activation=Swish
        )
        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            input_channel=in_channels, cfg=self.mobilevit_config["layer1"]
        )
        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            input_channel=in_channels, cfg=self.mobilevit_config["layer2"]
        )
        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            input_channel=in_channels, cfg=self.mobilevit_config["layer3"]
        )
        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=self.mobilevit_config["layer4"],
            dilate=False,
        )
        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            input_channel=in_channels,
            cfg=self.mobilevit_config["layer5"],
            dilate=False,
        )
        in_channels = out_channels
        exp_channels = min(self.mobilevit_config["last_layer_exp_factor"] * in_channels, 960)
        self.conv_1x1_exp = ConvNormActivation(
            in_planes=in_channels,
            out_planes=exp_channels,
            kernel_size=1,
            stride=1,
            activation=Swish
        )
        if 0.0 < classifier_dropout < 1.0:
            self.classifier = nn.SequentialCell(OrderedDict([
                ('global_pool', GlobalAvgPooling(keep_dims=False)),
                ('dropout', nn.Dropout(keep_prob=1 - classifier_dropout)),
                ('fc', nn.Dense(in_channels=exp_channels, out_channels=num_classes, has_bias=True)),
            ]))
        else:
            self.classifier = nn.SequentialCell(OrderedDict([
                ('global_pool', GlobalAvgPooling(keep_dims=False)),
                ('fc', nn.Dense(in_channels=exp_channels, out_channels=num_classes, has_bias=True)),
            ]))

    def _make_layer(self, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Tuple[nn.SequentialCell, int]:
        """
        Generate a layer with MobileNetv2block or MobileViTblock according to the configuration information

        Args:
           input_channel(int):Input channel
           cfg(dict):parameters for the corresponding model's specification
           dilate(bool):Set whether to dilate

        Returns:
            Tuple,a SequentialCell and out channel
        """
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                input_channel=input_channel, cfg=cfg, dilate=dilate
            )

        return self._make_mobilenet_layer(
            input_channel=input_channel, cfg=cfg
        )

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.SequentialCell, int]:
        """
        Generate a layer with MobileNetv2 block

        Args:
           input_channel(int):Input channel
           cfg(dict):parameters for the corresponding model's specification

        Returns:
            Tuple,a SequentialCell and out channel
        """
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []
        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channel=input_channel,
                out_channel=output_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
            block.append(layer)
            input_channel = output_channels
        return nn.SequentialCell(block), input_channel

    def _make_mit_layer(self, input_channel, cfg: Dict, dilate: Optional[bool] = False) \
            -> Tuple[nn.SequentialCell, int]:
        """
        Generate a layer with MobileViTBlock

        Args:
           input_channel(int):Input channel
           cfg(dict):parameters for the corresponding model's specification
           dilate(boo):Set whether to dilate

        Returns:
            Tuple,a SequentialCell and out channel
        """
        block = []
        stride = cfg.get("stride", 1)
        if stride == 2:
            if dilate:
                self.dilation *= 2
                stride = 1
            layer = InvertedResidual(
                in_channel=input_channel,
                out_channel=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
            )
            block.append(layer)
            input_channel = cfg.get("out_channels")
        head_dim = cfg.get("head_dim", 32)
        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        if head_dim is None:
            num_heads = cfg.get("num_heads", 4)
            if num_heads is None:
                num_heads = 4
            head_dim = transformer_dim // num_heads
        block.append(
            MobileViTBlock(
                in_channels=input_channel,
                transformer_dim=transformer_dim,
                ffn_dim=ffn_dim,
                n_transformer_blocks=cfg.get("transformer_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=0.05,
                ffn_dropout=0.0,
                attn_dropout=0.0,
                head_dim=head_dim,
                no_fusion=False,
                conv_ksize=3,
            )
        )
        return nn.SequentialCell(block), input_channel

    def construct(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        x = self.classifier(x)
        return x
