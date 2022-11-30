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
"""Loss Cell"""

from mindspore import nn


class CustomWithLossCell(nn.Cell):
    """
    Connecting the network to the loss function.

    Notes:
        The number of columns in the dataset is not 2, then the forward network
        needs to be connected to the loss function.

    Args:
        net (Cell): PFLDInference. Backbone Network.
        net_auxiliary (Cell): AuxiliaryNet. Auxiliary Network.
        loss_fn (Cell): PFLDLoss. Loss function.

    Outputs:
        loss value.
    """

    def __init__(self,
                 net: nn.Cell,
                 net_auxiliary: nn.Cell,
                 loss_fn: nn.Cell):
        super(CustomWithLossCell, self).__init__()
        self.net = net
        self.net_auxiliary = net_auxiliary
        self._loss_fn = loss_fn

    def construct(self, img, landmark_gt, weight_attribute, euler_angle):
        """ build network """
        feature1, landmark = self.net(img)
        angle = self.net_auxiliary(feature1)
        return self._loss_fn(angle,
                             landmark,
                             landmark_gt,
                             weight_attribute,
                             euler_angle)
