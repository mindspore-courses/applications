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
""" landmark head """

from mindspore import nn


class LandmarkHead(nn.Cell):
    """
    Network head with 68 or 98 feature points obtained from feature map.

    Args:
        feature_num (int): Dimension of last layer.
        landmark_num (int): Output dimension.

    Returns:
        Tensor, predicted landmarks.
    """

    def __init__(self,
                 feature_num: int = 176,
                 landmark_num: int = 68):
        super(LandmarkHead, self).__init__()
        self.fc = nn.Dense(feature_num, landmark_num * 2)

    def construct(self, x):
        """ build network """
        landmark = self.fc(x)
        return landmark
