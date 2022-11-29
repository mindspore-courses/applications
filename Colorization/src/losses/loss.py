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

import mindspore
import mindspore.numpy
import mindspore.nn as nn


class NetLoss(nn.Cell):
    """
    Connecting the network to the loss function.

    Args:
        net (nn.Cell): Colorization Network.

    Inputs:
        - **x** (tensor): Grayscale image.
        - **gt** (tensor): Color picture ground truth.
        - **weight** (numpy.array): Weight of each pixel.

    Outputs:
        loss value.
    """
    def __init__(self, net):
        super(NetLoss, self).__init__(auto_prefix=True)
        self.net = net
        self.loss = nn.CrossEntropyLoss(reduction='none')

    def construct(self, images, targets, boost, mask):
        """ build network """
        outputs = self.net(images)
        boost_nongray = boost * mask
        squeeze = mindspore.ops.Squeeze(1)
        boost_nongray = squeeze(boost_nongray)
        result = self.loss(outputs, targets)
        result_loss = (result * boost_nongray).mean()
        return result_loss
