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
"""Mixer module."""

from mindspore import nn
from mindspore.ops import operations as P

from mindvision.classification.models.blocks import FeedForward

class TokenMixer(nn.Cell):
    """
    Allow communication within patches - communication between different patches in the same channel.

    Args:
        hidden_dim (int): The hidden dimension of the model - number of channels.
        num_patches (int): The number of patches.
        token_mlp_dim (int): The tunable hidden widths in the token-mixing MLP.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = TokenMixer(hidden_dim=C, num_patch=S, token_mlp_dim=Ds)
    """
    def __init__(self,
                 hidden_dim: int,
                 num_patches: int,
                 token_mlp_dim: int,
                 keep_prob: float = 1.0):
        super(TokenMixer, self).__init__()
        self.norm = nn.LayerNorm((hidden_dim,))
        self.transpose = P.Transpose()
        self.token_mixing_mlp = FeedForward(in_features=num_patches,
                                            hidden_features=token_mlp_dim,
                                            activation=nn.GELU,
                                            keep_prob=keep_prob)
    def construct(self, x):
        """Token Mixer construct."""
        x = self.norm(x)
        x = self.transpose(x, (0, 2, 1))
        x = self.token_mixing_mlp(x)
        x = self.transpose(x, (0, 2, 1))
        return x

class ChannelMixer(nn.Cell):
    """
    Allow communication within channels - among channels.

    Args:
        hidden_dim (int): The hidden dimension of the model - number of channels.
        channel_mlp_dim (int): The tunable hidden widths in the channel-mixing MLP.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = ChannelMixer(hidden_dim=C, channel_mlp_dim=Dc)
    """
    def __init__(self,
                 hidden_dim: int,
                 channel_mlp_dim: int,
                 keep_prob: float = 1.0):
        super(ChannelMixer, self).__init__()
        self.norm: Optional[nn.Cell] = nn.LayerNorm((hidden_dim,))
        self.channel_mixing_mlp = FeedForward(in_features=hidden_dim,
                                              hidden_features=channel_mlp_dim,
                                              activation=nn.GELU,
                                              keep_prob=keep_prob)

    def construct(self, x):
        """Channel Mixer construct."""
        x = self.norm(x)
        x = self.channel_mixing_mlp(x)
        return x

class MixerBlock(nn.Cell):
    """
    The mixer block layer for the MLP-Mixer Architecture.

    Args:
        hidden_dim (int): The hidden dimension of the model - number of channels.
        num_patches (int): The number of patches.
        token_mlp_dim (int): The tunable hidden widths in the token-mixing MLP.
        channel_mlp_dim (int): The tunable hidden widths in the channel-mixing MLP.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = MixerBlock(hidden_dim=C, num_patches=S, token_mlp_dim=Ds, channel_mlp_dim=Dc)
    """
    def __init__(self,
                 hidden_dim: int,
                 num_patches: int,
                 token_mlp_dim: int,
                 channel_mlp_dim: int,
                 keep_prob: float = 1.0):
        super(MixerBlock, self).__init__()
        self.token_mixer = TokenMixer(hidden_dim=hidden_dim,
                                      num_patches=num_patches,
                                      token_mlp_dim=token_mlp_dim,
                                      keep_prob=keep_prob)
        self.channel_mixer = ChannelMixer(hidden_dim=hidden_dim,
                                          channel_mlp_dim=channel_mlp_dim,
                                          keep_prob=keep_prob)

    def construct(self, x):
        """Mixer Block construct."""
        x += self.token_mixer(x)
        x += self.channel_mixer(x)
        return x

class DeepMixerBlock(nn.Cell):
    """
    Muiti mixer block layer for the MLP-Mixer Architecture.

    Args:
        num_blocks (int): No of Blocks of MixerBlock to use - deep of network.
        hidden_dim (int): The hidden dimension of the model - number of channels.
        num_patches (int): The number of patches.
        token_mlp_dim (int): The tunable hidden widths in the token-mixing MLP.
        channel_mlp_dim (int): The tunable hidden widths in the channel-mixing MLP.
        keep_prob (float): The keep rate, greater than 0 and less equal than 1. Default: 1.0.

     Returns:
        Tensor, output tensor.

    Examples:
        >>> ops = MixerBlock(hidden_dim=C, num_patches=S, token_mlp_dim=Ds, channel_mlp_dim=Dc)
    """
    def __init__(self,
                 num_blocks: int = 12,
                 hidden_dim: int = 768,
                 num_patches: int = 196,
                 token_mlp_dim: int = 384,
                 channel_mlp_dim: int = 3072,
                 keep_prob: float = 1.0):
        super(DeepMixerBlock, self).__init__()
        layers = []
        for _ in range(num_blocks):
            layers.append(
                MixerBlock(hidden_dim=hidden_dim,
                           num_patches=num_patches,
                           token_mlp_dim=token_mlp_dim,
                           channel_mlp_dim=channel_mlp_dim,
                           keep_prob=keep_prob)
            )
        self.layers = nn.SequentialCell(layers)

    def construct(self, x):
        """Deep Mixer Block construct."""
        return self.layers(x)
