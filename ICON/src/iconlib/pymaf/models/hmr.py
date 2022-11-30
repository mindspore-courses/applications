"""hmr module"""
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
import mindspore.nn as nn


class Bottleneck(nn.Cell):
    """ Redefinition of Bottleneck residual block
        Adapted from the official PyTorch implementation
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=inplanes, out_channels=planes, kernel_size=1, has_bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=planes, momentum=0.9)
        self.conv2 = nn.Conv2d(
            in_channels=planes,
            out_channels=planes,
            kernel_size=3,
            stride=stride,
            pad_mode="pad",
            padding=1,
            has_bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_features=planes, momentum=0.9)
        self.conv3 = nn.Conv2d(
            in_channels=planes, out_channels=planes * 4, kernel_size=1, has_bias=False
        )
        self.bn3 = nn.BatchNorm2d(num_features=planes * 4, momentum=0.9)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        """construct"""
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResnetBackbone(nn.Cell):
    """ Feature Extractor with ResNet backbone
    """

    def __init__(self, model="res50"):
        if model == "res50":
            block, layers = Bottleneck, [3, 4, 6, 3]
        else:
            pass  # TODO

        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            pad_mode="pad",
            padding=3,
            has_bias=False,
        )
        self.bn1 = nn.BatchNorm2d(num_features=64, use_batch_statistics=True)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        """make layer"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                nn.Conv2d(
                    in_channels=self.inplanes,
                    out_channels=planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    has_bias=False,
                ),
                nn.BatchNorm2d(num_features=planes * block.expansion, momentum=0.9),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            print(i)
            layers.append(block(self.inplanes, planes))

        return nn.SequentialCell(*layers)

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        x_4 = self.layer4(x_3)

        x_f = self.avgpool(x_4)

        x_f = x_f.view(x_f.shape[0], -1)

        x_featmap = x_4

        return x_featmap, x_f
