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
"""metrics for nerf"""

import mindspore as md

__all__ = ["mse", "psnr_from_mse"]


def mse(img_1, img_2):
    """
    MSE between two images.

    Args:
        img_1 (Tensor): The first set of images.
        img_2 (Tensor): The second set of images.

    Returns:
        Tensor, MSE loss.
    """
    return md.numpy.mean((img_1 - img_2)**2)


def psnr_from_mse(mse_logit):
    """
    Convert MSE to PSNR.

    Args:
        mse_logit (Tensor): MSE loss value.

    Returns:
        Tensor, the PSNR value.
    """
    return -10.0 * (md.numpy.log(mse_logit) / md.numpy.log(md.Tensor([10.0])))
