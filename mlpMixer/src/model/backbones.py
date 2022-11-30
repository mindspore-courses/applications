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
"""Mixer backbones."""

from mindspore import nn
from mindspore.ops import operations as P

from mindvision.classification.models.blocks import PatchEmbedding

from model.mixer_blocks import DeepMixerBlock

class Mixer(nn.Cell):
    """
    MLP-Mixer Architecture implementation.

    Args:
        image_size (int): Input image size. Default: 224.
        input_channels (int): The number of input channel. Default: 3.
        patch_size (int): Patch size of image. Default: 16.
        num_blocks (int): Number of Blocks of MixerBlock to use.
        hidden_dim (int): The hidden dimension of the model - number of channels.
        token_mlp_dim (int): The tunable hidden widths in the token-mixing MLP.
        channel_mlp_dim (int): The tunable hidden widths in the channel-mixing MLP.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.

    Outputs:
        Model

    Examples:
        >>> ops = Mixer(image_size = 224,
                        input_channels = 3,
                        patch_size = 16,
                        num_blocks = 12,
                        hidden_dim = 768,
                        token_mlp_dim = 384,
                        channel_mlp_dim = 3072,
                        keep_prob = 1.0)
    """
    def __init__(self,
                 image_size: int = 224,
                 input_channels: int = 3,
                 patch_size: int = 16,
                 num_blocks: int = 12,
                 hidden_dim: int = 768,
                 token_mlp_dim: int = 384,
                 channel_mlp_dim: int = 3072,
                 keep_prob: float = 1.0):
        super(Mixer, self).__init__()
        self.patch_embedding = PatchEmbedding(image_size=image_size,
                                              patch_size=patch_size,
                                              embed_dim=hidden_dim,
                                              input_channels=input_channels)
        self.num_patches = self.patch_embedding.num_patches
        self.mixer_blocks = DeepMixerBlock(num_blocks=num_blocks,
                                           hidden_dim=hidden_dim,
                                           num_patches=self.num_patches,
                                           token_mlp_dim=token_mlp_dim,
                                           channel_mlp_dim=channel_mlp_dim,
                                           keep_prob=keep_prob)
        self.norm = nn.LayerNorm((hidden_dim,))
        self.avg_global_pool = P.ReduceMean(keep_dims=False)

    def construct(self, x):
        """MLP-Mixer construct."""
        x = self.patch_embedding(x)
        x = self.mixer_blocks(x)
        x = self.norm(x)
        x = self.avg_global_pool(x, 1)
        return x
