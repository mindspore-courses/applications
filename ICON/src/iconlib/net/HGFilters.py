"""hg filter"""
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
from mindspore import nn, ops

from iconlib.net.net_util import ConvBlock


class HourGlass(nn.Cell):
    """hour gclass"""
    def __init__(self, num_modules, depth, num_features, opt):
        super(HourGlass, self).__init__()
        self.num_modules = num_modules
        self.depth = depth
        self.features = num_features
        self.opt = opt

        self._generate_network(self.depth)

    def _generate_network(self, level):
        """_generate_network"""
        self.insert_child_to_cell(
            "b1_" + str(level), ConvBlock(self.features, self.features, self.opt)
        )

        self.insert_child_to_cell(
            "b2_" + str(level), ConvBlock(self.features, self.features, self.opt)
        )

        if level > 1:
            self._generate_network(level - 1)
        else:
            self.insert_child_to_cell(
                "b2_plus_" + str(level),
                ConvBlock(self.features, self.features, self.opt),
            )

        self.insert_child_to_cell(
            "b3_" + str(level), ConvBlock(self.features, self.features, self.opt)
        )

    def _construct(self, level, inp):
        """_construct"""
        # Upper branch
        up1 = inp
        layer = getattr(self, "b1_" + str(level))
        up1 = layer(up1)

        # Lower branch
        low1 = ops.avg_pool2d(inp, 2, strides=2)
        layer = getattr(self, "b2_" + str(level))
        low1 = layer(low1)

        if level > 1:
            low2 = self._construct(level - 1, low1)
        else:
            low2 = low1
            layer = getattr(self, "b2_plus_" + str(level))
            low2 = layer(low2)

        low3 = low2
        layer = getattr(self, "b3_" + str(level))
        low3 = layer(low3)

        up2 = F.interpolate(low3, scale_factor=2, mode="bicubic", align_corners=True)
        # up2 = F.interpolate(low3, scale_factor=2, mode='nearest)

        return up1 + up2

    def construct(self, x):
        return self._forward(self.depth, x)


class HGFilter(nn.Cell):
    """HG filter"""
    def __init__(self, opt, num_modules, in_dim):
        super(HGFilter, self).__init__()

        self.num_modules = num_modules

        self.opt = opt

        [k, s, d, p] = self.opt.conv1

        self.conv1 = nn.Conv2d(
            in_channels=in_dim,
            out_channels=64,
            kernel_size=k,
            stride=s,
            dilation=d,
            has_bias=True,
            pad_mode="pad",
            padding=p,
        )

        if self.opt.norm == "batch":
            self.bn1 = nn.BatchNorm2d(64)
        elif self.opt.norm == "group":
            self.bn1 = nn.GroupNorm(32, 64)

        if self.opt.hg_down == "conv64":
            self.conv2 = ConvBlock(64, 64, self.opt)
            self.down_conv2 = nn.Conv2d(
                64, 128, kernel_size=3, stride=2, pad_mode="pad", padding=1
            )
        elif self.opt.hg_down == "conv128":
            self.conv2 = ConvBlock(64, 128, self.opt)
            self.down_conv2 = nn.Conv2d(
                128, 128, kernel_size=3, stride=2, pad_mode="pad", padding=1
            )
        elif self.opt.hg_down == "ave_pool":
            self.conv2 = ConvBlock(64, 128, self.opt)
        else:
            raise NameError("Unknown Fan Filter setting!")

        self.conv3 = ConvBlock(128, 128, self.opt)
        self.conv4 = ConvBlock(128, 256, self.opt)

        # Stacking part
        for hg_module in range(self.num_modules):
            self.insert_child_to_cell(
                "m" + str(hg_module), HourGlass(1, opt.num_hourglass, 256, self.opt)
            )

            self.insert_child_to_cell(
                "top_m_" + str(hg_module), ConvBlock(256, 256, self.opt)
            )
            self.insert_child_to_cell(
                "conv_last" + str(hg_module),
                nn.Conv2d(
                    256,
                    256,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    pad_mode="pad",
                    has_bias=True,
                ),
            )
            if self.opt.norm == "batch":
                self.insert_child_to_cell(
                    "bn_end" + str(hg_module), nn.BatchNorm2d(256)
                )
            elif self.opt.norm == "group":
                self.insert_child_to_cell(
                    "bn_end" + str(hg_module), nn.GroupNorm(32, 256)
                )

            self.insert_child_to_cell(
                "l" + str(hg_module),
                nn.Conv2d(
                    256,
                    opt.hourglass_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    pad_mode="pad",
                    has_bias=True,
                ),
            )

            if hg_module < self.num_modules - 1:
                self.insert_child_to_cell(
                    "bl" + str(hg_module),
                    nn.Conv2d(
                        256,
                        256,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        pad_mode="pad",
                        has_bias=True,
                    ),
                )
                self.insert_child_to_cell(
                    "al" + str(hg_module),
                    nn.Conv2d(
                        opt.hourglass_dim,
                        256,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        pad_mode="pad",
                        has_bias=True,
                    ),
                )

    def construct(self, x):
        """construct"""
        relu = ops.ReLU()
        x = relu(self.bn1(self.conv1(x)), True)
        if self.opt.hg_down == "ave_pool":
            x = ops.avg_pool2d(self.conv2(x), 2, strides=2)
        elif self.opt.hg_down in ["conv64", "conv128"]:
            x = self.conv2(x)
            x = self.down_conv2(x)
        else:
            raise NameError("Unknown Fan Filter setting!")

        x = self.conv3(x)
        x = self.conv4(x)

        previous = x

        outputs = []
        for i in range(self.num_modules):
            layer = getattr(self, "m" + str(i))
            h_g = layer(previous)

            l_l = h_g
            layer = getattr(self, "top_m_" + str(i))
            l_l = layer(l_l)

            layer1 = getattr(self, "bn_end" + str(i))
            layer2 = getattr(self, "conv_last" + str(i))
            l_l = relu(layer1(layer2(l_l)), True)

            # Predict heatmaps
            layer = getattr(self, "l" + str(i))
            tmp_out = layer(l_l)
            outputs.append(tmp_out)

            if i < self.num_modules - 1:
                layer = getattr(self, "bl" + str(i))
                l_l = layer(l_l)
                layer = getattr(self, "al" + str(i))
                tmp_out_ = layer(tmp_out)
                previous = previous + l_l + tmp_out_

        return outputs
