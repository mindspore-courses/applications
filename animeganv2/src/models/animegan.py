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
"""The connector of animegan network, loss and optimizer."""

import mindspore.nn as nn


class AnimeGAN(nn.Cell):
    """
    Connect all networks.

    Args:
        train_one_step_cell_d (Cell): Discriminator network training package class.
        train_one_step_cell_g (Cell): Generator network training package class.

    Inputs:
        - **img** (tensor) - Real world image.
        - **anime** (tensor) - Original anime image.
        - **anime_gray** (tensor) - Gray anime image.
        - **anime_smt_gray** (tensor) - Smoothed gray anime image.

    Outputs:
        - **d_loss** (tensor) - Discriminator loss.
        - **g_loss** (tensor) - Generator loss.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> animegan = AnimeGAN()
    """

    def __init__(self, train_one_step_cell_d, train_one_step_cell_g):
        super(AnimeGAN, self).__init__(auto_prefix=True)
        self.train_one_step_cell_g = train_one_step_cell_g
        self.train_one_step_cell_d = train_one_step_cell_d

    def construct(self, img, anime, anime_gray, anime_smt_gray):
        """ build network """
        output_d_loss = self.train_one_step_cell_d(img, anime, anime_gray, anime_smt_gray)
        output_g_loss = self.train_one_step_cell_g(img, anime_gray)
        return output_d_loss, output_g_loss
