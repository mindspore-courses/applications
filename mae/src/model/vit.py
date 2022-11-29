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
# =========================================================================
"""
Vit.
"""

import numpy as np

from mindspore import nn
from mindspore import Tensor
from mindspore import ops as P
from mindspore import dtype as mstype
from mindspore.common.parameter import Parameter
from mindspore.nn.transformer import TransformerEncoder
from mindspore.train.serialization import load_param_into_net
from mindspore.common.initializer import initializer, XavierUniform

from model.modules import PatchEmbed
from model.mae_vit import MAEModule
from utils.helper import get_2d_sin_cos_pos_embed


class Vit(MAEModule):
    """
    Vit module.

    Args:
        batch_size (int): Batch size.
        patch_size (int): Patch size.
        image_size (int): Image size.
        encoder_layers (int): Number of encoder layer. Default: 12.
        encoder_num_heads (int): Number of encoder head. Default: 12.
        encoder_dim (int): Encoder dimension. Default: 768.
        mlp_ratio (int): Multiplicative between encoder dimension and fully connected layer. Default: 4.
        channels (int): Channel number. Default: 3.
        dropout (float): Neuron discard probability. Default: 0.
        drop_path (float): Percentage of drop path. Default: 0.1.
        global_pool (bool): Whether use global pool. Default: True.
        initialization (initializer): Parameter initialization method.

    Inputs:
        -**img**  (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        -**out** (Tensor),The tensor after vit model.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>>Vit(batch_size=32, patch_size=16, image_size=224, encoder_layers=12, encoder_num_heads=12,
        ...    encoder_dim=768, mlp_ratio=4, channels=3, dropout=0., drop_path=0.1)
    """
    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 encoder_layers=12,
                 encoder_num_heads=12,
                 encoder_dim=768,
                 mlp_ratio=4,
                 channels=3,
                 dropout=0.,
                 drop_path=0.1,
                 global_pool=True,
                 initialization=XavierUniform()):
        super(Vit, self).__init__(batch_size, image_size, patch_size)
        cls_token = Parameter(
            initializer(initialization, (1, 1, encoder_dim)),
            name='cls', requires_grad=True
        )

        # the number and shape of copying cls_token into (batch_size, 1, 1)
        self.cls_token = P.Tile()(cls_token, (batch_size, 1, 1))
        seq_length = self.num_patches + 1

        # position embedding
        self.encoder_pos_embedding = Parameter(
            initializer(initialization, (1, seq_length, encoder_dim)),
            name='pos_embedding', requires_grad=False
        )

        # Encoder module in Transformer with multiple stacked TransformerEncoderLayers,
        # including multi-headed self-attentive layers and feedforward layers.
        self.encoder = TransformerEncoder(batch_size=batch_size,
                                          num_layers=encoder_layers,
                                          num_heads=encoder_num_heads,
                                          hidden_size=encoder_dim,
                                          ffn_hidden_size=encoder_dim*mlp_ratio,
                                          seq_length=seq_length,
                                          hidden_dropout_rate=drop_path)
        self.add = P.Add()
        self.cast = P.Cast()
        self.cat = P.Concat(axis=1)
        self.norm = nn.LayerNorm((encoder_dim,), epsilon=1e-5).to_float(mstype.float32)
        self.fc_norm = nn.LayerNorm((encoder_dim,), epsilon=1e-5).to_float(mstype.float32)
        self.global_pool = global_pool
        self.reduce_mean = P.ReduceMean()

        # Embed each patch and output an n-dimensional vector to represent the patch.
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size,
                                      in_features=channels, out_features=encoder_dim)
        if dropout:
            # Regularization, randomly setting some neuron outputs to 0 during training
            self.is_dropout = True
            self.dropout = nn.Dropout(keep_prob=(1. - dropout))
        self.encoder_input_mask = Tensor(np.ones((batch_size, seq_length, seq_length)), mstype.float32)
        self._init_weights()

    def _init_weights(self):

        # Initialize weights, using two-dimensional sin-cos position encoding
        encoder_pos_emd = Tensor(
            get_2d_sin_cos_pos_embed(self.encoder_pos_embedding.shape[-1],
                                     int(self.num_patches ** .5),
                                     cls_token=True),
            mstype.float32
        )
        self.encoder_pos_embedding.set_data(P.ExpandDims()(encoder_pos_emd, 0))

    def construct(self, img):
        """build network"""
        tokens = self.patch_embed(img)
        tokens = self.cat((self.cls_token, tokens))
        tokens = self.add(tokens, self.encoder_pos_embedding)

        if self.is_dropout:
            # Need to perform dropout operation
            temp = self.cast(tokens, mstype.float32)
            temp = self.dropout(temp)
            tokens = self.cast(temp, tokens.dtype)

        # Input tokens to Encoder
        tokens = self.encoder(tokens, self.encoder_input_mask)[0]

        if self.global_pool:
            # Global pooling needs to be performed
            token = tokens[:, 1:, :]
            tokens = self.reduce_mean(token, 1)
            out = self.fc_norm(tokens)
        else:
            tokens = self.norm(tokens)
            out = tokens[:, 0]

        return out


class FineTuneVit(nn.Cell):
    """
    FineTuneViT module.

    Args:
        batch_size (int): Batch size.
        patch_size (int): Patch size.
        image_size (int): Image size.
        num_classes (int): number of classes. Default: 1000.
        dropout (float): Neuron discard probability. Default: 0.
        drop_path (float): Percentage of drop path. Default: 0.1.
        initialization (initializer): Parameter initialization method.
        encoder_dim (int): The dimension of encoder. Default=768.
        encoder_layers (int): The number of layer of encoder. Default=12.
        encoder_num_heads (int) The number of head of encoder. Default=12.
        mlp_ratio (int): Multiple relationship between encoder dimension and full connection layer. Default=4.
        channels (int): The number of image channel. Default=3.

    Inputs:
        -**img**  (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        -**head(tokens)** (Tensor),The tensor after vit model and full connection layer.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>>net = FineTuneVit(batch_size=32, patch_size=16, image_size=224, dropout=0.0,
        ...                  num_classes=1000, encoder_layers=12,encoder_num_heads=12, encoder_dim=768,
        ...                  mlp_ratio=4, drop_path=0.1, channels=3)
    """
    def __init__(self,
                 batch_size,
                 patch_size,
                 image_size,
                 num_classes=1000,
                 dropout=0.,
                 drop_path=0.1,
                 initialization=XavierUniform(),
                 encoder_dim=768,
                 encoder_layers=12,
                 encoder_num_heads=12,
                 mlp_ratio=4,
                 channels=3):
        super(FineTuneVit, self).__init__()
        self.encoder = Vit(batch_size, patch_size, image_size, encoder_layers, encoder_num_heads, encoder_dim,
                           mlp_ratio, channels, dropout, drop_path)
        encoder_dim = encoder_dim
        self.head = nn.Dense(encoder_dim, num_classes)
        self.head.weight.set_data(initializer(initialization, [num_classes, encoder_dim]))

    def init_weights(self, param_dict):
        net_not_load = load_param_into_net(self, param_dict)
        return net_not_load

    @staticmethod
    def no_weight_decay():
        return {'encoder.cls_token', 'encoder.encoder_pos_embedding'}

    def construct(self, img):
        """ build network """
        tokens = self.encoder(img)
        return self.head(tokens)
