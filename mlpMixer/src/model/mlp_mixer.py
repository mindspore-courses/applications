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

"""MLP Mixer Model."""

from typing import Optional
import ml_collections as collections

from mindvision.classification.models.classifiers import BaseClassifier
from mindvision.classification.models.head import DenseHead, MultilayerDenseHead

from model.backbones import Mixer

def mixer(image_size: int,
          input_channels: int,
          patch_size: int,
          embed_dim: int,
          num_layers: int,
          token_mlp_dim: int,
          channel_mlp_dim: int,
          num_classes: int,
          arch,
          dropout: float = 0.,
          representation_size: Optional[int] = None) -> Mixer:
    """MLP-Mixer architecture."""
    backbone = Mixer(image_size=image_size,
                     input_channels=input_channels,
                     patch_size=patch_size,
                     num_blocks=num_layers,
                     hidden_dim=embed_dim,
                     token_mlp_dim=token_mlp_dim,
                     channel_mlp_dim=channel_mlp_dim,
                     keep_prob=1.0 - dropout)
    if representation_size:
        head = MultilayerDenseHead(input_channel=embed_dim,
                                   num_classes=num_classes,
                                   mid_channel=[representation_size],
                                   activation=['tanh', None],
                                   keep_prob=[1.0, 1.0])
    else:
        head = DenseHead(input_channel=embed_dim,
                         num_classes=num_classes)
    model = BaseClassifier(backbone=backbone, head=head)
    print(arch)
    return model

def mixer_b_16(num_classes: int = 10,
               image_size: int = 224,
               has_logits: bool = False,
               drop_out: float = 0.0
              ) -> Mixer:
    """
    Constructs a Mixer_b_16 architecture from
    `MLP-Mixer: An all-MLP Architecture for Vision <https://arxiv.org/abs/2107.08391>`_.

    Args:
        image_size (int): The input image size. Default: 224.
        num_classes (int): The number of classification. Default: 10.
        has_logits (bool): Whether has logits or not. Default: False.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        drop_out (float): The drop out rate. Default: 0.0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import Mixer_l_16
        >>>
        >>> net = mixer_b_16()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About MLP-Mixer:

    The MLP-Mixer architecture (or “Mixer” for short) is an image architecture
    that doesn't use convolutions or self-attention.
    Instead, Mixer’s architecture is based entirely on multi-layer perceptrons (MLPs) that
    are repeatedly applied across either spatial locations or feature channels.
    Mixer relies only on basic matrix multiplication routines,
    changes to data layout (reshapes and transpositions), and scalar nonlinearities.

    It accepts a sequence of linearly projected image patches (also referred to as tokens) shaped as
    a “patches × channels” table as an input, and maintains this dimensionality.
    Mixer makes use of two types of MLP layers: channel-mixing MLPs and token-mixing MLPs.
    The channel-mixing MLPs allow communication between different channels;
    they operate on each token independently and take individual rows of the table as inputs.
    The token-mixing MLPs allow communication between different spatial locations (tokens);
    they operate on each channel independently and take individual columns of the table as inputs.
    These two types of layers are interleaved to enable interaction of both input dimensions.

    Citation:

    .. code-block::

    @article{tolstikhin2021mlp,
      title={Mlp-mixer: An all-mlp architecture for vision},
      author={Tolstikhin, Ilya O and Houlsby, Neil and Kolesnikov, Alexander and Beyer,
      Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Steiner,
      Andreas and Keysers, Daniel and Uszkoreit, Jakob and others},
      journal={Advances in Neural Information Processing Systems},
      volume={34},
      year={2021}
    }
    """

    config = collections.ConfigDict()
    config.arch = 'Mixer_b_16_' + str(image_size)
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 16
    config.embed_dim = 768
    config.token_mlp_dim = 384
    config.channel_mlp_dim = 3072
    config.num_layers = 12
    config.dropout = drop_out
    config.input_channels = 3
    config.representation_size = 1024 if has_logits else None

    return mixer(**config)

def mixer_l_16(num_classes: int = 10,
               image_size: int = 224,
               has_logits: bool = False,
               drop_out: float = 0.0
              ) -> Mixer:
    """
    Constructs a Mixer_l_16 architecture from
    `MLP-Mixer: An all-MLP Architecture for Vision <https://arxiv.org/abs/2107.08391>`_.

    Args:
        image_size (int): The input image size. Default: 224.
        num_classes (int): The number of classification. Default: 10.
        has_logits (bool): Whether has logits or not. Default: False.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        drop_out (float): The drop out rate. Default: 0.0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import Mixer_l_16
        >>>
        >>> net = mixer_l_16()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About MLP-Mixer:

    The MLP-Mixer architecture (or “Mixer” for short) is an image architecture
    that doesn't use convolutions or self-attention.
    Instead, Mixer’s architecture is based entirely on multi-layer perceptrons (MLPs) that
    are repeatedly applied across either spatial locations or feature channels.
    Mixer relies only on basic matrix multiplication routines,
    changes to data layout (reshapes and transpositions), and scalar nonlinearities.

    It accepts a sequence of linearly projected image patches (also referred to as tokens) shaped as
    a “patches × channels” table as an input, and maintains this dimensionality.
    Mixer makes use of two types of MLP layers: channel-mixing MLPs and token-mixing MLPs.
    The channel-mixing MLPs allow communication between different channels;
    they operate on each token independently and take individual rows of the table as inputs.
    The token-mixing MLPs allow communication between different spatial locations (tokens);
    they operate on each channel independently and take individual columns of the table as inputs.
    These two types of layers are interleaved to enable interaction of both input dimensions.

    Citation:

    .. code-block::

    @article{tolstikhin2021mlp,
      title={Mlp-mixer: An all-mlp architecture for vision},
      author={Tolstikhin, Ilya O and Houlsby, Neil and Kolesnikov, Alexander and Beyer,
      Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Steiner,
      Andreas and Keysers, Daniel and Uszkoreit, Jakob and others},
      journal={Advances in Neural Information Processing Systems},
      volume={34},
      year={2021}
    }
    """

    config = collections.ConfigDict()
    config.arch = 'Mixer_l_16_' + str(image_size)
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 16
    config.embed_dim = 1024
    config.token_mlp_dim = 512
    config.channel_mlp_dim = 4096
    config.num_layers = 24
    config.dropout = drop_out
    config.input_channels = 3
    config.representation_size = 1024 if has_logits else None

    return mixer(**config)

def mixer_b_32(num_classes: int = 10,
               image_size: int = 224,
               has_logits: bool = False,
               drop_out: float = 0.0
              ) -> Mixer:
    """
    Constructs a Mixer_b_32 architecture from
    `MLP-Mixer: An all-MLP Architecture for Vision <https://arxiv.org/abs/2107.08391>`_.

    Args:
        image_size (int): The input image size. Default: 224.
        num_classes (int): The number of classification. Default: 10.
        has_logits (bool): Whether has logits or not. Default: False.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        drop_out (float): The drop out rate. Default: 0.0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import Mixer_b_32
        >>>
        >>> net = mixer_b_32()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About MLP-Mixer:

    The MLP-Mixer architecture (or “Mixer” for short) is an image architecture
    that doesn't use convolutions or self-attention.
    Instead, Mixer’s architecture is based entirely on multi-layer perceptrons (MLPs) that
    are repeatedly applied across either spatial locations or feature channels.
    Mixer relies only on basic matrix multiplication routines,
    changes to data layout (reshapes and transpositions), and scalar nonlinearities.

    It accepts a sequence of linearly projected image patches (also referred to as tokens) shaped as
    a “patches × channels” table as an input, and maintains this dimensionality.
    Mixer makes use of two types of MLP layers: channel-mixing MLPs and token-mixing MLPs.
    The channel-mixing MLPs allow communication between different channels;
    they operate on each token independently and take individual rows of the table as inputs.
    The token-mixing MLPs allow communication between different spatial locations (tokens);
    they operate on each channel independently and take individual columns of the table as inputs.
    These two types of layers are interleaved to enable interaction of both input dimensions.

    Citation:

    .. code-block::

    @article{tolstikhin2021mlp,
      title={Mlp-mixer: An all-mlp architecture for vision},
      author={Tolstikhin, Ilya O and Houlsby, Neil and Kolesnikov,
      Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung,
      Jessica and Steiner, Andreas and Keysers, Daniel and Uszkoreit, Jakob and others},
      journal={Advances in Neural Information Processing Systems},
      volume={34},
      year={2021}
    }
    """

    config = collections.ConfigDict()
    config.arch = 'Mixer_b_32_' + str(image_size)
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 32
    config.embed_dim = 768
    config.token_mlp_dim = 384
    config.channel_mlp_dim = 3072
    config.num_layers = 12
    config.dropout = drop_out
    config.input_channels = 3
    config.representation_size = 1024 if has_logits else None

    return mixer(**config)

def mixer_s_16(num_classes: int = 10,
               image_size: int = 224,
               has_logits: bool = False,
               drop_out: float = 0.0
              ) -> Mixer:
    """
    Constructs a Mixer_b_16 architecture from
    `MLP-Mixer: An all-MLP Architecture for Vision <https://arxiv.org/abs/2107.08391>`_.

    Args:
        image_size (int): The input image size. Default: 224.
        num_classes (int): The number of classification. Default: 10.
        has_logits (bool): Whether has logits or not. Default: False.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        drop_out (float): The drop out rate. Default: 0.0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import Mixer_s_16
        >>>
        >>> net = mixer_s_16()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About MLP-Mixer:

    The MLP-Mixer architecture (or “Mixer” for short) is an image architecture
    that doesn't use convolutions or self-attention.
    Instead, Mixer’s architecture is based entirely on multi-layer perceptrons (MLPs) that
    are repeatedly applied across either spatial locations or feature channels.
    Mixer relies only on basic matrix multiplication routines,
    changes to data layout (reshapes and transpositions), and scalar nonlinearities.

    It accepts a sequence of linearly projected image patches (also referred to as tokens) shaped as
    a “patches × channels” table as an input, and maintains this dimensionality.
    Mixer makes use of two types of MLP layers: channel-mixing MLPs and token-mixing MLPs.
    The channel-mixing MLPs allow communication between different channels;
    they operate on each token independently and take individual rows of the table as inputs.
    The token-mixing MLPs allow communication between different spatial locations (tokens);
    they operate on each channel independently and take individual columns of the table as inputs.
    These two types of layers are interleaved to enable interaction of both input dimensions.

    Citation:

    .. code-block::

    @article{tolstikhin2021mlp,
      title={Mlp-mixer: An all-mlp architecture for vision},
      author={Tolstikhin, Ilya O and Houlsby, Neil and Kolesnikov, Alexander and Beyer,
      Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung, Jessica and Steiner,
      Andreas and Keysers, Daniel and Uszkoreit, Jakob and others},
      journal={Advances in Neural Information Processing Systems},
      volume={34},
      year={2021}
    }
    """

    config = collections.ConfigDict()
    config.arch = 'Mixer_s_16_' + str(image_size)
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 16
    config.embed_dim = 512
    config.token_mlp_dim = 256
    config.channel_mlp_dim = 2048
    config.num_layers = 8
    config.dropout = drop_out
    config.input_channels = 3
    config.representation_size = 1024 if has_logits else None

    return mixer(**config)


def mixer_s_8(num_classes: int = 10,
              image_size: int = 224,
              has_logits: bool = False,
              drop_out: float = 0.0
             ) -> Mixer:
    """
    Constructs a Mixer_b_16 architecture from
    `MLP-Mixer: An all-MLP Architecture for Vision <https://arxiv.org/abs/2107.08391>`_.

    Args:
        image_size (int): The input image size. Default: 224.
        num_classes (int): The number of classification. Default: 10.
        has_logits (bool): Whether has logits or not. Default: False.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        drop_out (float): The drop out rate. Default: 0.0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import Mixer_s_8
        >>>
        >>> net = mixer_s_8()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About MLP-Mixer:

    The MLP-Mixer architecture (or “Mixer” for short) is an image architecture
    that doesn't use convolutions or self-attention.
    Instead, Mixer’s architecture is based entirely on multi-layer perceptrons (MLPs) that
    are repeatedly applied across either spatial locations or feature channels.
    Mixer relies only on basic matrix multiplication routines,
    changes to data layout (reshapes and transpositions), and scalar nonlinearities.

    It accepts a sequence of linearly projected image patches (also referred to as tokens) shaped as
    a “patches × channels” table as an input, and maintains this dimensionality.
    Mixer makes use of two types of MLP layers: channel-mixing MLPs and token-mixing MLPs.
    The channel-mixing MLPs allow communication between different channels;
    they operate on each token independently and take individual rows of the table as inputs.
    The token-mixing MLPs allow communication between different spatial locations (tokens);
    they operate on each channel independently and take individual columns of the table as inputs.
    These two types of layers are interleaved to enable interaction of both input dimensions.

    Citation:

    .. code-block::

    @article{tolstikhin2021mlp,
      title={Mlp-mixer: An all-mlp architecture for vision},
      author={Tolstikhin, Ilya O and Houlsby, Neil and Kolesnikov,
      Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner, Thomas and Yung,
      Jessica and Steiner, Andreas and Keysers, Daniel and Uszkoreit, Jakob and others},
      journal={Advances in Neural Information Processing Systems},
      volume={34},
      year={2021}
    }
    """

    config = collections.ConfigDict()
    config.arch = 'Mixer_s_16_' + str(image_size)
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 8
    config.embed_dim = 512
    config.token_mlp_dim = 256
    config.channel_mlp_dim = 2048
    config.num_layers = 8
    config.dropout = drop_out
    config.input_channels = 3
    config.representation_size = 1024 if has_logits else None

    return mixer(**config)

def mixer_s_32(num_classes: int = 10,
               image_size: int = 224,
               has_logits: bool = False,
               drop_out: float = 0.0
              ) -> Mixer:
    """
    Constructs a Mixer_b_16 architecture from
    `MLP-Mixer: An all-MLP Architecture for Vision <https://arxiv.org/abs/2107.08391>`_.

    Args:
        image_size (int): The input image size. Default: 224.
        num_classes (int): The number of classification. Default: 10.
        has_logits (bool): Whether has logits or not. Default: False.
        pretrained (bool): Whether to download and load the pre-trained model. Default: False.
        drop_out (float): The drop out rate. Default: 0.0.

    Inputs:
        - **x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        Tensor of shape :math:`(N, CLASSES_{out})`

    Supported Platforms:
        ``GPU``

    Examples:
        >>> import numpy as np
        >>>
        >>> import mindspore as ms
        >>> from mindvision.classification.models import Mixer_s_32
        >>>
        >>> net = mixer_s_32()
        >>> x = ms.Tensor(np.ones([1, 3, 224, 224]), ms.float32)
        >>> output = net(x)
        >>> print(output.shape)
        (1, 1000)

    About MLP-Mixer:

    The MLP-Mixer architecture (or “Mixer” for short) is an image architecture
    that doesn't use convolutions or self-attention.
    Instead, Mixer’s architecture is based entirely on multi-layer perceptrons (MLPs) that
    are repeatedly applied across either spatial locations or feature channels.
    Mixer relies only on basic matrix multiplication routines,
    changes to data layout (reshapes and transpositions), and scalar nonlinearities.

    It accepts a sequence of linearly projected image patches (also referred to as tokens) shaped as
    a “patches × channels” table as an input, and maintains this dimensionality.
    Mixer makes use of two types of MLP layers: channel-mixing MLPs and token-mixing MLPs.
    The channel-mixing MLPs allow communication between different channels;
    they operate on each token independently and take individual rows of the table as inputs.
    The token-mixing MLPs allow communication between different spatial locations (tokens);
    they operate on each channel independently and take individual columns of the table as inputs.
    These two types of layers are interleaved to enable interaction of both input dimensions.

    Citation:

    .. code-block::

    @article{tolstikhin2021mlp,
      title={Mlp-mixer: An all-mlp architecture for vision},
      author={Tolstikhin, Ilya O and Houlsby, Neil and Kolesnikov,
      Alexander and Beyer, Lucas and Zhai, Xiaohua and Unterthiner,
      Thomas and Yung, Jessica and Steiner,
      Andreas and Keysers, Daniel and Uszkoreit, Jakob and others},
      journal={Advances in Neural Information Processing Systems},
      volume={34},
      year={2021}
    }
    """

    config = collections.ConfigDict()
    config.arch = 'Mixer_s_32_' + str(image_size)
    config.image_size = image_size
    config.num_classes = num_classes
    config.patch_size = 32
    config.embed_dim = 512
    config.token_mlp_dim = 256
    config.channel_mlp_dim = 2048
    config.num_layers = 8
    config.dropout = drop_out
    config.input_channels = 3
    config.representation_size = 1024 if has_logits else None

    return mixer(**config)
