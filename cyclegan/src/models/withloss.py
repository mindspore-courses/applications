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
"""Define with losses."""

import mindspore.nn as nn


class WithLossCell(nn.Cell):
    """
    Wrap the network with loss function to return generator loss.

    Args:
        network (Cell): The target network to wrap.
    """
    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, img_a, img_b):
        _, _, lg, _, _, _, _, _, _ = self.network(img_a, img_b)
        return lg
