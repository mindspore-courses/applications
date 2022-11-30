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
"""MobilenetV1 backbone."""
import mindspore.nn as nn
from mindspore.ops import operations as P


def conv_bn_relu(in_channel, out_channel, kernel_size, stride, depthwise, activation='relu6'):
    """
    Convolution batchnormalization and ReLU activatition.

    Args:
        in_channels (int): The channel number of the input tensor of the Conv2d layer.
        out_channels (int): The channel number of the output tensor of the Conv2d layer.
        kernel_size (Union[int, tuple[int]]): Specifies the height and width of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the height
            and width of the convolution kernel. A tuple of two integers represents the height
            and width of the convolution kernel respectively. Default: 3.
        stride (Union[int, tuple[int]]): The movement stride of the 2D convolution kernel.
            The data type is an integer or a tuple of two integers. An integer represents the movement step size
            in both height and width directions. A tuple of two integers represents the movement step size in the height
            and width directions respectively. Default: 1.
        depthwise (bool): Whether calculate gradient through depth.
        activation (str): The activation function. Default: 'relu6'.

    Returns:
        SequentialCell, Combine several layers toghter.
    """
    output = []
    output.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, pad_mode='same',
                            group=1 if not depthwise else in_channel))
    output.append(nn.BatchNorm2d(out_channel))
    if activation:
        output.append(nn.get_activation(activation))
    return nn.SequentialCell(output)


class MobileNetV1(nn.Cell):
    """
    MobileNet V1 backbone

    Args:
        class_num (int): Class number. Default: 1001.
        features_only (bool): Whether to choose only feature. Default: False.

    Inputs:
        - **x** (Tensor) - Input block.

    Outputs:
        Tensor, output block.

    Support Plarforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> mobilenet_v1 = MobileNetV1(class_num=1000, features_only=True)
    """
    def __init__(self, class_num=1001, features_only=False):
        super(MobileNetV1, self).__init__()
        self.features_only = features_only
        cnn = [
            conv_bn_relu(3, 32, 3, 2, False),

            conv_bn_relu(32, 32, 3, 1, True),
            conv_bn_relu(32, 64, 1, 1, False),

            conv_bn_relu(64, 64, 3, 2, True),
            conv_bn_relu(64, 128, 1, 1, False),
            conv_bn_relu(128, 128, 3, 1, True),
            conv_bn_relu(128, 128, 1, 1, False),

            conv_bn_relu(128, 128, 3, 2, True),
            conv_bn_relu(128, 256, 1, 1, False),
            conv_bn_relu(256, 256, 3, 1, True),
            conv_bn_relu(256, 256, 1, 1, False),

            conv_bn_relu(256, 256, 3, 2, True),
            conv_bn_relu(256, 512, 1, 1, False),
            conv_bn_relu(512, 512, 3, 1, True),
            conv_bn_relu(512, 512, 1, 1, False),
            conv_bn_relu(512, 512, 3, 1, True),
            conv_bn_relu(512, 512, 1, 1, False),
            conv_bn_relu(512, 512, 3, 1, True),
            conv_bn_relu(512, 512, 1, 1, False),
            conv_bn_relu(512, 512, 3, 1, True),
            conv_bn_relu(512, 512, 1, 1, False),
            conv_bn_relu(512, 512, 3, 1, True),
            conv_bn_relu(512, 512, 1, 1, False),

            conv_bn_relu(512, 512, 3, 2, True),
            conv_bn_relu(512, 1024, 1, 1, False),
            conv_bn_relu(1024, 1024, 3, 1, True),
            conv_bn_relu(1024, 1024, 1, 1, False),
        ]

        if self.features_only:
            self.network = nn.CellList(cnn)
        else:
            self.network = nn.SequentialCell(cnn)
            self.fc = nn.Dense(1024, class_num)

    def construct(self, x):
        """Construct Mobilenetv1 backbone"""
        output = x
        if self.features_only:
            features = ()
            for block in self.network:
                output = block(output)
                features = features + (output,)
            return features

        output = self.network(x)
        output = P.ReduceMean()(output, (2, 3))
        output = self.fc(output)
        return output


class FeatureSelector(nn.Cell):
    """
    Select specific layers from en entire feature list.

    Args:
        feature_idxes(List): Feature indexes.

    Inputs:
        - **feature_list** (List) - A list of features.

    Outputs:
        List, selected features.

    Support Plarforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> selector = FeatureSelector([6, 10, 22, 26])
    """
    def __init__(self, feature_idxes):
        super(FeatureSelector, self).__init__()
        self.feature_idxes = feature_idxes

    def construct(self, feature_list):
        """Construct Feature Selector"""
        selected = ()
        for i in self.feature_idxes:
            selected = selected + (feature_list[i],)
        return selected


class MobileNetV1FeatureSelector(nn.Cell):
    """
    Mobilenet v1 feature selector.

    Args:
        class_num (int): Class number. Default: 1001.
        features_only (bool): Whether to choose only feature. Default: False.

    Inputs:
        - **x** (Cell) - Input block.

    Outputs:
        Cell, output block.

    Support Plarforms:
        ``Ascend`` ``CPU`` ``GPU``

    Examples:
        >>> mobilenet_v1_fea_selector = MobileNetV1FeatureSelector(class_num=1000, features_only=True)
    """
    def __init__(self, class_num=1001, features_only=False):
        super(MobileNetV1FeatureSelector, self).__init__()
        self.mobilenet_v1 = MobileNetV1(class_num=class_num, features_only=features_only)
        self.selector = FeatureSelector([6, 10, 22, 26])

    def construct(self, x):
        """Construct Mobilenet v1 feature selector"""
        features = self.mobilenet_v1(x)
        features = self.selector(features)
        return features
