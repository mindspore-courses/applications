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
"""RetinaFace heads."""
import math

from mindspore import nn, ops

from utils.initialize import init_kaiming_uniform


class ClassHead(nn.Cell):
    """
    RetinaFace ClassHead, judge whether anchor contains face.
    Args:
        inchannels(int): Channel of input tensor. Default: 512.
        num_anchors(int): Number of anchors. Default: 3.

    Inputs:
        - **x** (Tensor) - Tensor from ssh, its channel number should be same with the argument inchannels.

    Outputs:
        A tensor, which means classification result of anchors, its shape is :math:`(B, H * W, num_anchors * 2)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> from src.model.head import ClassHead
        >>> head = ClassHead(2, 1)
        >>> zeros = ops.Zeros()
        >>> x = zeros((1, 2, 2, 2), ms.float32)
        >>> x = head(x)
        >>> print(x)
        [[[-0.44703963  0.08408954]
          [-0.44703963  0.08408954]
          [-0.44703963  0.08408954]
          [-0.44703963  0.08408954]]]
    """

    def __init__(self, inchannels=512, num_anchors=3):
        super(ClassHead, self).__init__()
        self.num_anchors = num_anchors

        weight_shape = (self.num_anchors * 2, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = init_kaiming_uniform(weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, self.num_anchors * 2, kernel_size=(1, 1), stride=1, padding=0,
                                 has_bias=True, weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """Forward pass."""
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (ops.Shape()(out)[0], -1, 2))


class BboxHead(nn.Cell):
    """
    RetinaFace BoxHead, predict height,width and center position of boxes.
    Args:
        inchannels(int): Channel of input tensor. Default: 512.
        num_anchors(int): Number of anchors. Default: 3.

    Inputs:
        - **x** (Tensor) - Tensor from ssh, which channel number is same with the inchannels argument.

    Outputs:
        A tensor, which means center position,width and height of boxes, its shape is
        :math:`(B, H * W, num_anchors * 4)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> from src.model.head import BboxHead
        >>> head = BboxHead(2, 1)
        >>> zeros = ops.Zeros()
        >>> x = zeros((1, 2, 2, 2), ms.float32)
        >>> x = head(x)
        >>> print(x)
        [[[ 0.69234294  0.680479   -0.5804417  -0.6484875 ]
          [ 0.69234294  0.680479   -0.5804417  -0.6484875 ]
          [ 0.69234294  0.680479   -0.5804417  -0.6484875 ]
          [ 0.69234294  0.680479   -0.5804417  -0.6484875 ]]]
    """

    def __init__(self, inchannels=512, num_anchors=3):
        super(BboxHead, self).__init__()

        weight_shape = (num_anchors * 4, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = init_kaiming_uniform(weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 4, kernel_size=(1, 1), stride=1, padding=0, has_bias=True,
                                 weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """Forward pass."""
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (ops.Shape()(out)[0], -1, 4))


class LandmarkHead(nn.Cell):
    """
    RetinaFace BoxHead, predict position of landmarks.

    Args:
        inchannels(int): Channel of input tensor. Default: 512.
        num_anchors(int): Number of anchors. Default: 3.

    Inputs:
        - **x** (Tensor) - Tensor from ssh, its channel number should be same with the argument inchannels.

    Outputs:
        A tensor, which means center position of 5 landmark points, its shape is :math:`(B, H * W, num_anchors * 10)`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import mindspore as ms
        >>> from mindspore import ops
        >>> from src.model.head import LandmarkHead
        >>> head = LandmarkHead(2, 1)
        >>> zeros = ops.Zeros()
        >>> x = zeros((1, 2, 2, 2), ms.float32)
        >>> x = head(x)
        >>> print(x.shape)
        (1, 4, 10)
    """

    def __init__(self, inchannels=512, num_anchors=3):
        super(LandmarkHead, self).__init__()

        weight_shape = (num_anchors * 10, inchannels, 1, 1)
        kaiming_weight, kaiming_bias = init_kaiming_uniform(weight_shape, a=math.sqrt(5), has_bias=True)
        self.conv1x1 = nn.Conv2d(inchannels, num_anchors * 10, kernel_size=(1, 1), stride=1, padding=0, has_bias=True,
                                 weight_init=kaiming_weight, bias_init=kaiming_bias)

        self.permute = ops.Transpose()
        self.reshape = ops.Reshape()

    def construct(self, x):
        """Forward pass."""
        out = self.conv1x1(x)
        out = self.permute(out, (0, 2, 3, 1))
        return self.reshape(out, (ops.Shape()(out)[0], -1, 10))
