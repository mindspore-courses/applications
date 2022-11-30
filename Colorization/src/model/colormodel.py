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
"""Define colorization networks"""

import mindspore.nn as nn

class ColorModel(nn.Cell):
    """
       Connect all networks.

    Args:
        myTrainOneStepCellForNet (nn.Cell): network training package class.

    Inputs:
        - **x** (tensor): Grayscale image.
        - **gt** (tensor): Color picture ground truth.
        - **weight** (numpy.array): Weight of each pixel.

    OutPuts:
         Loss value.

    Examples:
           >>> colormodel = ColorModel()
    """
    def __init__(self, my_train_one_step_cell_for_net):
        super(ColorModel, self).__init__(auto_prefix=True)
        self.my_train_one_step_cell_for_net = my_train_one_step_cell_for_net

    def construct(self, result, targets, boost, mask):
        loss = self.my_train_one_step_cell_for_net(result, targets, boost, mask)
        return loss
