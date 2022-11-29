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
""" PFLD auxiliary network """

from mindspore import nn

from mindvision.classification.models.blocks import ConvNormActivation


class AuxiliaryNet(nn.Cell):
    """
    PFLD auxiliary network definition.

    Args:
       channel_num (tuple): Where the meaning of the element is the number of input channels per network module.

    Returns:
        Tensor, output tensor, the shape is (batch_size * 3).
    """

    def __init__(self,
                 channel_num: tuple = (64, 128, 128, 32, 128, 32)):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = ConvNormActivation(channel_num[0], channel_num[1], 3, 2)
        self.conv2 = ConvNormActivation(channel_num[1], channel_num[2], 3, 1)
        self.conv3 = ConvNormActivation(channel_num[2], channel_num[3], 3, 2)
        self.conv4 = ConvNormActivation(channel_num[3], channel_num[4], 2, 2)

        self.max_pool1 = nn.MaxPool2d(3, 2)
        self.fc1 = nn.Dense(channel_num[4], channel_num[5])
        self.fc2 = nn.Dense(channel_num[5], 3)

    def construct(self, x):
        """ build network """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view((x.shape[0], -1))
        x = self.fc1(x)
        x = self.fc2(x)
        return x
