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
"""Define Unet Skip Connection Block."""

import mindspore.nn as nn
import mindspore.ops as ops


class UNetSkipConnectionBlock(nn.Cell):
    """
    Unet submodule with skip connection.

    Args:
        outer_nc (int): The number of filters in the outer conv layer.
        inner_nc (int): The number of filters in the inner conv layer.
        in_planes (int): The number of channels in input images/features.
        dropout (bool): Use dropout or not. Default: False.
        submodule (Cell): Previously defined submodules.
        outermost (bool): If this module is the outermost module.
        innermost (bool): If this module is the innermost module.
        alpha (float): LeakyRelu slope. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".

    Outputs:
        Tensor, output tensor of Unet submodule.
    """

    def __init__(self, outer_nc, inner_nc, in_planes=None, dropout=False,
                 submodule=None, outermost=False, innermost=False, alpha=0.2, norm_mode='batch'):
        super(UNetSkipConnectionBlock, self).__init__()
        downnorm = nn.BatchNorm2d(inner_nc)
        upnorm = nn.BatchNorm2d(outer_nc)
        use_bias = False
        if norm_mode == 'instance':
            downnorm = nn.BatchNorm2d(inner_nc, affine=False)
            upnorm = nn.BatchNorm2d(outer_nc, affine=False)
            use_bias = True
        if in_planes is None:
            in_planes = outer_nc
        downconv = nn.Conv2d(in_planes, inner_nc, kernel_size=4,
                             stride=2, padding=1, has_bias=use_bias, pad_mode='pad')
        downrelu = nn.LeakyReLU(alpha)
        uprelu = nn.ReLU()

        if outermost:
            upconv = nn.Conv2dTranspose(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, pad_mode='pad')
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.Conv2dTranspose(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, has_bias=use_bias, pad_mode='pad')
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.Conv2dTranspose(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, has_bias=use_bias, pad_mode='pad')
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up
            if dropout:
                model.append(nn.Dropout(0.5))

        self.model = nn.SequentialCell(model)
        self.skip_connections = not outermost
        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        out = self.model(x)
        if self.skip_connections:
            out = self.concat((out, x))
        return out
