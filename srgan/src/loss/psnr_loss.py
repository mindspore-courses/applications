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
"""PSNRLOSS"""

import mindspore.nn as nn

__all__ = ["PSNRLoss"]

class PSNRLoss(nn.Cell):
    """
    Loss for SRResnet.

    Args:
        generator (nn.Cell): SRResnet.

    Inputs:
        - **hr_img** (Tensor) - The high-resolution image.
          The input shape must be (batchsize, num_channels, height, width).
        - **lr_img** (Tensor) - The low-resolution image.
          The input shape must be (batchsize, num_channels, height, width).

    Outputs:
        - **psnr_loss** (Tensor) - The pixel-wise PSNR(MSE) loss.
          The output has the shape (batchsize, loss_value).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> import mindspore.nn as nn
        >>> from src.model.generator import get_generator
        >>> generator = get_generator(4, 0.02)
        >>> psnr_loss = PSNRLoss(generator)
        >>> hr_img = Tensor(np.zeros([16, 3, 96, 96]),mstype.float32)
        >>> lr_img = Tensor(np.zeros([16, 3, 24, 24]),mstype.float32)
        >>> loss_value = psnr_loss(hr_img, lr_img)
        >>> print(loss_value)
        0.0
    """
    def __init__(self, generator):
        super(PSNRLoss, self).__init__()
        self.generator = generator
        self.pixel_criterion = nn.MSELoss()

    def construct(self, hr_img, lr_img):
        hr = hr_img
        sr = self.generator(lr_img)
        psnr_loss = self.pixel_criterion(hr, sr)
        return psnr_loss
