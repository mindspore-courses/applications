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
"""MobileViT_module."""

from collections import OrderedDict
from typing import Optional, Dict, Tuple

from mindspore import nn, Tensor, ops
from mindspore import numpy as np

from models.blocks.convnormactivation import ConvNormActivation
from models.swish import Swish
from models.transform import TransformerEncoder


class MobileViTBlock(nn.Cell):
    """
    This class defines the MobileViT module
    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (Optional[int]): Number of transformer blocks. Default: None
        head_dim (Optional[int]): Head dimension in the multi-head attention. Default: None
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: None
        dropout (Optional[float]): Dropout rate. Default: None
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: None
        patch_h (Optional[int]): Patch height for unfolding operation. Default: None
        patch_w (Optional[int]): Patch width for unfolding operation. Default: None
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: None
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: None
    """

    def __init__(
            self,
            in_channels: int,
            transformer_dim: int,
            ffn_dim: int,
            n_transformer_blocks: Optional[int] = None,
            head_dim: Optional[int] = None,
            attn_dropout: Optional[float] = None,
            dropout: Optional[float] = None,
            ffn_dropout: Optional[float] = None,
            patch_h: Optional[int] = None,
            patch_w: Optional[int] = None,
            conv_ksize: Optional[int] = None,
            no_fusion: Optional[bool] = None,
    ) -> None:

        conv_1x1_out = ConvNormActivation(
            in_planes=transformer_dim,
            out_planes=in_channels,
            kernel_size=1,
            stride=1,
            activation=Swish
        )
        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = ConvNormActivation(
                in_planes=2 * in_channels,
                out_planes=in_channels,
                kernel_size=conv_ksize,
                stride=1,
                activation=Swish
            )

        super(MobileViTBlock, self).__init__()
        self.local_rep = nn.SequentialCell(OrderedDict([
            ('conv_3x3',
             ConvNormActivation(in_planes=in_channels, out_planes=in_channels, kernel_size=conv_ksize, stride=1,
                                activation=Swish)),
            ('conv_1x1', ConvNormActivation(in_planes=in_channels, out_planes=transformer_dim,
                                            kernel_size=1, stride=1, norm=None, activation=None)),
        ]))
        num_heads = transformer_dim // head_dim
        global_rep = [
            TransformerEncoder(
                dim=transformer_dim,
                mlp_dim=ffn_dim,
                num_heads=num_heads,
                num_layers=n_transformer_blocks,
                attention_keep_prob=1 - attn_dropout,
                keep_prob=1 - dropout,
                drop_path_keep_prob=1 - ffn_dropout,
                activation=Swish
            )
        ]
        norm_layer = nn.LayerNorm((transformer_dim,))
        global_rep.append(norm_layer)

        self.global_rep = nn.SequentialCell(global_rep)
        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h
        self.resize_unfolding = nn.ResizeBilinear()
        self.resize_folding = nn.ResizeBilinear()
        self.concat_op = ops.Concat(axis=1)

    def unfolding(self, feature_map: Tensor) -> Tuple[Tensor, Dict]:
        """
        Reshape the input Tensor into a series of flattened blocks and project it into a fixed d-dimensional space

        Args:
            feature_map(Tensor):input feature map tensor

        Returns:
            Tuple,reshaped patches and properties of patches
        """
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = int(patch_w * patch_h)
        batch_size, in_channels, orig_h, orig_w = feature_map.shape
        new_h = int(np.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(np.ceil(orig_w / self.patch_w) * self.patch_w)
        interpolate = False

        # Note: Padding can be done, but then it needs to be handled in attention function.
        if new_w != orig_w or new_h != orig_h:
            feature_map = self.resize_unfolding(feature_map, size=(new_h, new_w), align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] --> [B * C * n_h, p_h, n_w, p_w]
        reshaped_fm = feature_map.reshape(
            batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w
        )

        # [B * C * n_h, p_h, n_w, p_w] --> [B * C * n_h, n_w, p_h, p_w]
        transposed_fm = ops.transpose(reshaped_fm, (0, 2, 1, 3))

        # [B * C * n_h, n_w, p_h, p_w] --> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = transposed_fm.reshape(
            batch_size, in_channels, num_patches, patch_area
        )

        # [B, C, N, P] --> [B, P, N, C]
        transposed_fm = ops.transpose(reshaped_fm, (0, 3, 2, 1))

        # [B, P, N, C] --> [BP, N, C]
        patches = transposed_fm.reshape(batch_size * patch_area, num_patches, -1)
        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }
        return patches, info_dict

    def folding(self, patches: Tensor, info_dict: Dict) -> Tensor:
        """
        Fold the tensors according to the order of the patches and the spatial order of the pixels inside the patches

        Args:
            patches(Tensor):input patches tensor
            info_dict(dict):the properties of patches

        Returns:
            Tensor,The folded feature map tensor
        """

        # [BP, N, C] --> [B, P, N, C]
        patches = patches.view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )
        batch_size = patches.shape[0]
        channels = patches.shape[3]
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] --> [B, C, N, P]
        patches = ops.transpose(patches, (0, 3, 2, 1))

        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = patches.reshape(
            batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w
        )

        # [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = ops.transpose(feature_map, (0, 2, 1, 3))

        # [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        feature_map = feature_map.reshape(
            batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w
        )
        if info_dict["interpolate"]:
            feature_map = self.resize_folding(feature_map, size=info_dict["orig_size"], align_corners=False)
        return feature_map

    def construct(self, x):
        res = x
        x = self.local_rep(x)
        patches, info_dict = self.unfolding(x)
        patches = self.global_rep(patches)
        x = self.folding(patches=patches, info_dict=info_dict)
        x = self.conv_proj(x)
        if self.fusion is not None:
            x = self.fusion(self.concat_op((res, x)))
        return x
