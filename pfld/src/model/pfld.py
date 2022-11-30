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
# ==============================================================================
""" PFLD net """

from mindspore import nn

from .backbone import PFLDBackbone
from .landmark_head import LandmarkHead


class PFLDBase(nn.Cell):
    """
    PFLD 1X version, which is suit for 68 landmark points.

    Args:
        channel_num (tuple): With this parameter, to define the input channel and output
            channel of every layer.
        feature_num (int): Dimension of last layer.
        landmark_num (int): Output dimension.

    Inputs:
        -**x** (Tensor) - Tensor of shape :math:`(N, C_{in}, H_{in}, W_{in})`.

    Outputs:
        -**features** (Tensor), used as input of auxiliary network.
        -**landmark** (Tensor), store the coordinate of landmark.

    Examples:
        >>> PFLDBase(channel_num=(3, 64, 64, 64, 64, 128, 128, 128, 16, 32, 128),
        ...          feature_num=176,
        ...          landmark_num=98)
    """
    def __init__(self,
                 channel_num: tuple = (3, 64, 64, 64, 64, 128, 128, 128, 16, 32, 128),
                 feature_num: int = 176,
                 landmark_num: int = 68):
        super(PFLDBase, self).__init__()

        self.backbone = PFLDBackbone(channel_num=channel_num)
        self.head = LandmarkHead(feature_num=feature_num, landmark_num=landmark_num)

    def construct(self, x):
        """ build network """
        features, multi_scale = self.backbone(x)
        landmark = self.head(multi_scale)
        return features, landmark


def pfld_1x_98():
    """
    PFLD 1X version, which is suit for 98 landmark points.

    Returns:
        PFLDBase. One version of PFLD Network.
    """

    return PFLDBase(channel_num=(3, 64, 64, 64, 64, 128, 128, 128, 16, 32, 128),
                    feature_num=176,
                    landmark_num=98)


def pfld_1x_68():
    """
    PFLD 1X version, which is suit for 68 landmark points.

    Returns:
        PFLDBase. One version of PFLD Network.
    """

    return PFLDBase(channel_num=(3, 64, 64, 64, 64, 128, 128, 128, 16, 32, 128),
                    feature_num=176,
                    landmark_num=68)
