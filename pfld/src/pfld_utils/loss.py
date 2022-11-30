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
# ===========================================================================
""" Calculation of PFLD losses based on attribute weights. """

from mindspore import nn
from mindspore import ops


class PFLDLoss(nn.Cell):
    """
    Constructor for PFLDLoss

    Inputs:
        -**angle** (float): Predict angle.
        -**landmark** (numpy.array): Predict landmark coordinate.
        -**landmark_gt** (numpy.array): Landmark coordinate ground truth.
        -**euler_angle_gt** (float): Angle ground truth.

    Outputs:
        Loss value.
    """

    def __init__(self):
        super(PFLDLoss, self).__init__()
        self.sum = ops.ReduceSum(keep_dims=False)
        self.cos = ops.Cos()
        self.mean = ops.ReduceMean(keep_dims=False)

    def construct(
            self,
            angle,
            landmark,
            landmark_gt,
            weight_attribute,
            euler_angle_gt):
        """Constructing the forward calculation process."""
        weight_angle = self.sum(1 - self.cos(angle - euler_angle_gt), 1)
        l2_distant = self.sum((landmark_gt - landmark) *
                              (landmark_gt - landmark), 1)
        loss = weight_angle * weight_attribute * l2_distant
        return self.mean(loss)
