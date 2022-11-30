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
"""Squeeze Excite Module."""

import mindspore.nn as nn
from mindspore import Tensor


class Swish(nn.Cell):
    """
    swish activation function.

    Args:
        None

    Return:
        Tensor

    Example:
        >>> x = Tensor(((20, 16), (50, 50)), mindspore.float32)
        >>> Swish()(x)
    """

    def __init__(self) -> None:
        super(Swish, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def construct(self, x) -> Tensor:
        """Swish construct."""
        return x * self.sigmoid(x)
