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
""" Define Pix2Pix model."""

import mindspore.nn as nn
from mindspore.common import initializer as init

from .generator import UNetGenerator
from .discriminator import Discriminator


def get_generator(config):
    """
    Return a generator by args.

    Args:
        config (class): Option class.

    Returns:
        net_generator. initialization generator network.
    """

    net_generator = UNetGenerator(in_planes=config.g_in_planes, out_planes=config.g_out_planes,
                                  ngf=config.g_ngf, n_layers=config.g_layers)
    for _, cell in net_generator.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if config.init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(config.init_gain), cell.weight.shape))
            elif config.init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(config.init_gain), cell.weight.shape))
            elif config.init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % config.init_type)
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))
    return net_generator

def get_discriminator(config):
    """
    Return a discriminator by args.

    Args:
        config (class): Option class.

     Returns:
        net_discriminator. initialization discriminator network.
    """

    net_discriminator = Discriminator(config, in_planes=config.d_in_planes, ndf=config.d_ndf,
                                      alpha=config.alpha, n_layers=config.d_layers)
    for _, cell in net_discriminator.cells_and_names():
        if isinstance(cell, (nn.Conv2d, nn.Conv2dTranspose)):
            if config.init_type == 'normal':
                cell.weight.set_data(init.initializer(init.Normal(config.init_gain), cell.weight.shape))
            elif config.init_type == 'xavier':
                cell.weight.set_data(init.initializer(init.XavierUniform(config.init_gain), cell.weight.shape))
            elif config.init_type == 'constant':
                cell.weight.set_data(init.initializer(0.001, cell.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % config.init_type)
        elif isinstance(cell, nn.BatchNorm2d):
            cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
            cell.beta.set_data(init.initializer('zeros', cell.beta.shape))
    return net_discriminator


class Pix2Pix(nn.Cell):
    """
    pix2pix model network.

    Args:
        discriminator (Cell): a generator.
        generator (Cell): a discriminator.

    Inputs:
        -**reala** - generate real image information.

    Outputs:
        fakeb, a fake image information.
    """
    def __init__(self, discriminator, generator):
        super(Pix2Pix, self).__init__(auto_prefix=True)
        self.net_discriminator = discriminator
        self.net_generator = generator

    def construct(self, reala):
        fakeb = self.net_generator(reala)
        return fakeb
