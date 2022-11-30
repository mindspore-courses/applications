"""FB network"""
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
import functools
from mindspore import nn


def get_norm_layer(norm_type="instance"):
    """get norm layer"""
    if norm_type == "batch":
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == "instance":
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)
    return norm_layer


def define_g(
        input_nc,
        output_nc,
        ngf,
        net_g,
        n_downsample_global=3,
        n_blocks_global=9,
        norm="instance",
        last_op=nn.Tanh(),
):
    """define network G"""
    norm_layer = get_norm_layer(norm_type=norm)
    if net_g == "global":
        net_g = GlobalGenerator(
            input_nc,
            output_nc,
            ngf,
            n_downsample_global,
            n_blocks_global,
            norm_layer,
            last_op=last_op,
        )

    return net_g


##############################################################################
# Generator
##############################################################################


class GlobalGenerator(nn.Cell):
    """global generator"""
    def __init__(
            self,
            input_nc,
            output_nc,
            ngf=64,
            n_downsampling=3,
            n_blocks=9,
            norm_layer=nn.BatchNorm2d,
            padding_type="reflect",
            last_op=nn.Tanh(),
    ):
        assert n_blocks >= 0
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU()

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                input_nc, ngf, kernel_size=7, padding=0, pad_mode="pad", has_bias=True
            ),
            norm_layer(ngf),
            activation,
        ]
        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [
                nn.Conv2d(
                    ngf * mult,
                    ngf * mult * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    pad_mode="pad",
                    has_bias=True,
                ),
                norm_layer(ngf * mult * 2),
                activation,
            ]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [
                ResnetBlock(
                    ngf * mult,
                    padding_type=padding_type,
                    activation=activation,
                    norm_layer=norm_layer,
                )
            ]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [
                nn.Conv2dTranspose(
                    ngf * mult,
                    int(ngf * mult / 2),
                    kernel_size=3,
                    stride=2,
                    has_bias=True,
                ),
                norm_layer(int(ngf * mult / 2)),
                activation,
            ]
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(
                ngf, output_nc, kernel_size=7, padding=0, pad_mode="pad", has_bias=True
            ),
        ]
        if last_op is not None:
            model += [last_op]

        self.model = nn.SequentialCell(*model)

    def construct(self, input_p):
        """construct"""
        return self.model(input_p)


# Define a resnet block
class ResnetBlock(nn.Cell):
    """resnet block"""
    def __init__(
            self, dim, padding_type, norm_layer, activation=nn.ReLU(), use_dropout=False
    ):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, activation, use_dropout
        )

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        """build conv block"""
        conv_block = []
        p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block += [
            nn.Conv2d(
                dim, dim, kernel_size=3, padding=p, pad_mode="pad", has_bias=True
            ),
            norm_layer(dim),
            activation,
        ]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p_p = 0
        if padding_type == "reflect":
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == "zero":
            p_p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)
        conv_block += [
            nn.Conv2d(
                dim, dim, kernel_size=3, padding=p_p, pad_mode="pad", has_bias=True
            ),
            norm_layer(dim),
        ]

        return nn.SequentialCell(*conv_block)

    def construct(self, x):
        """construct"""
        out = x + self.conv_block(x)
        return out


if __name__ == "__main__":
    net = define_g(6, 3, 64, "global", 4, 9, "instance")
