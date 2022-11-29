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
# ==============================================================================
"""upsample"""
import mindspore.nn as nn

from src.model.blocks import ConvBlock


class UpsampleUnit(nn.Cell):
    """ Upsample Unit for MSPN

    Args:
        ind (int): The order index of UpsampleUnit in UpsampleModule.
        in_channels (int): Input Tensor Channels.
        upsample_size (tuple): Upsample Tensor Shape.
        output_channel_num (int): Output Tensor Channels.
        output_shape (tuple): Output Tensor Shape.
        channel_num (int): Interim Tensor Channels. Default: 256.
        generate_skip (bool): Whether to generate skip connection output tensor. Default: False.
        generate_cross_conv (bool): Whether to apply Convolution for the output Tensor to further feed into next stage \
        Module. Default: False.

    Inputs:
        - **x** (Tensor) - A tensor to be inputted into upsample unit, which should have four dimensions with data \
        type float32.
        - **up_x** (Tensor) - A tensor from previous upsample unit, which should have four dimensions with data type \
        float32.

    Outputs:
        Tuple of 5 Tensor, the processed tensors.

        - **out** (Tensor) - A tensor which is the output of upsample unit, has the same dimension and data type as `x`.
        - **res** (Tensor) - A tensor which is the output of upsample unit and used for feature aggregation, has the \
        same dimension and data type as `x`.
        - **skip_tensor_1** (Tensor) - A tensor which is the output of the upsample unit, has the same dimension \
        and data type as `x`.
        - **skip_tensor_2** (Tensor) - A tensor which is the output of the upsample unit, has the same dimension \
        and data type as `x`.
        - **cross_conv** (Tensor) - A tensor which is used for cross stage feature aggregation, has the same dimension \
        and data type as `x`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> upsample_unit = UpsampleUnit(0, 3, (128, 128), 16, (256, 256), channel_num=256, generate_skip=False)
    """
    def __init__(self,
                 ind: int,
                 in_channels: int,
                 upsample_size: tuple,
                 output_channel_num: int,
                 output_shape: tuple,
                 channel_num: int = 256,
                 generate_skip: bool = False,
                 generate_cross_conv: bool = False
                 ) -> None:
        super(UpsampleUnit, self).__init__()
        self.output_shape = output_shape
        self.resize_bilinear = nn.ResizeBilinear()
        self.u_skip = ConvBlock(in_channels, channel_num, kernel_size=1, stride=1, padding=0, use_bn=True,
                                use_relu=False)
        self.relu = nn.ReLU()

        self.ind = ind
        if self.ind > 0:
            self.upsample_size = upsample_size
            self.up_conv = ConvBlock(channel_num, channel_num, kernel_size=1, stride=1, padding=0, use_bn=True,
                                     use_relu=False)

        self.generate_skip = generate_skip
        if self.generate_skip:
            self.skip1 = ConvBlock(in_channels, in_channels, kernel_size=1, stride=1, padding=0, use_bn=True,
                                   use_relu=True)
            self.skip2 = ConvBlock(channel_num, in_channels, kernel_size=1, stride=1, padding=0, use_bn=True,
                                   use_relu=True)

        self.generate_cross_conv = generate_cross_conv
        if self.ind == 3 and self.generate_cross_conv:
            self.cross_conv = ConvBlock(channel_num, 64, kernel_size=1, stride=1, padding=0, use_bn=True,
                                        use_relu=True)

        self.res_conv1 = ConvBlock(channel_num, channel_num, kernel_size=1, stride=1, padding=0, use_bn=True,
                                   use_relu=True)
        self.res_conv2 = ConvBlock(channel_num, output_channel_num, kernel_size=3, stride=1, padding=1, use_bn=True,
                                   use_relu=False)

    def construct(self, x, up_x):
        """Construct Func"""
        out = self.u_skip(x)

        if self.ind > 0:
            up_x = self.resize_bilinear(up_x, size=self.upsample_size, align_corners=True)
            up_x = self.up_conv(up_x)
            out += up_x
        out = self.relu(out)

        res = self.res_conv1(out)
        res = self.res_conv2(res)
        res = self.resize_bilinear(res, size=self.output_shape, align_corners=True)

        skip_tensor_1 = None
        skip_tensor_2 = None
        if self.generate_skip:
            skip_tensor_1 = self.skip1(x)
            skip_tensor_2 = self.skip2(out)

        cross_conv = None
        if self.ind == 3 and self.generate_cross_conv:
            cross_conv = self.cross_conv(out)

        return out, res, skip_tensor_1, skip_tensor_2, cross_conv


class UpsampleModule(nn.Cell):
    """ Upsample Module for MSPN

    Args:
        output_channel_num (int): Output Tensor Channels.
        output_shape (tuple): Output Tensor Shape.
        channel_num (int): Interim Tensor Channels. Default: 256.
        generate_skip (bool): Whether to generate skip connection output tensor. Default: False.
        generate_cross_conv (bool): Whether to apply Convolution for the output Tensor to further feed into next stage \
        Module. Default: False.

    Inputs:
        - **x4** (Tensor) - A tensor which is the output of the first downsample sub-module, which should have four \
        dimensions with data type float32
        - **x3** (Tensor) - A tensor which is the output of the second downsample sub-module, which should have four \
        dimensions with data type float32
        - **x2** (Tensor) - A tensor which is the output of the third downsample sub-module, which should have four \
        dimensions with data type float32
        - **x1** (Tensor) - A tensor which is the output of the fourth downsample sub-module, which should have four \
        dimensions with data type float32

    Outputs:
        Tuple of 4 Tensor, the processed tensors.

        - **res** (Tensor) - A tensor which is the output of upsample module and used for feature aggregation, has the \
        same dimension and data type as `x1`.
        - **skip_tensor_1** (Tensor) - A tensor which is the output of the upsample module, has the same dimension \
        and data type as `x1`.
        - **skip_tensor_2** (Tensor) - A tensor which is the output of the upsample module, has the same dimension \
        and data type as `x1`.
        - **cross_conv** (Tensor) - A tensor which is used for cross stage feature aggregation, has the same dimension \
        and data type as `x1`.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> upsample = UpsampleModule(64, (256, 256), channel_num=256, generate_skip=False, generate_cross_conv=False)
    """
    def __init__(self,
                 output_channel_num: int,
                 output_shape: tuple,
                 channel_num: int = 256,
                 generate_skip: bool = False,
                 generate_cross_conv: bool = False
                 ) -> None:
        super(UpsampleModule, self).__init__()
        self.in_channels = [2048, 1024, 512, 256]
        h, w = output_shape
        self.upsample_sizes = [(h // 8, w // 8), (h // 4, w // 4), (h // 2, w // 2), (h, w)]
        self.generate_skip = generate_skip
        self.generate_cross_conv = generate_cross_conv

        self.up1 = UpsampleUnit(0, self.in_channels[0], self.upsample_sizes[0], output_channel_num=output_channel_num,
                                output_shape=output_shape, channel_num=channel_num, generate_skip=self.generate_skip,
                                generate_cross_conv=self.generate_cross_conv)
        self.up2 = UpsampleUnit(1, self.in_channels[1], self.upsample_sizes[1], output_channel_num=output_channel_num,
                                output_shape=output_shape, channel_num=channel_num, generate_skip=self.generate_skip,
                                generate_cross_conv=self.generate_cross_conv)
        self.up3 = UpsampleUnit(2, self.in_channels[2], self.upsample_sizes[2], output_channel_num=output_channel_num,
                                output_shape=output_shape, channel_num=channel_num, generate_skip=self.generate_skip,
                                generate_cross_conv=self.generate_cross_conv)
        self.up4 = UpsampleUnit(3, self.in_channels[3], self.upsample_sizes[3], output_channel_num=output_channel_num,
                                output_shape=output_shape, channel_num=channel_num, generate_skip=self.generate_skip,
                                generate_cross_conv=self.generate_cross_conv)

    def construct(self, x4, x3, x2, x1):
        """Construct Func"""
        out1, res1, skip1_1, skip2_1, _ = self.up1(x4, None)
        out2, res2, skip1_2, skip2_2, _ = self.up2(x3, out1)
        out3, res3, skip1_3, skip2_3, _ = self.up3(x2, out2)
        _, res4, skip1_4, skip2_4, cross_conv = self.up4(x1, out3)

        # 'res' starts from small size
        res = [res1, res2, res3, res4]
        skip_tensor_1 = [skip1_4, skip1_3, skip1_2, skip1_1]
        skip_tensor_2 = [skip2_4, skip2_3, skip2_2, skip2_1]

        return res, skip_tensor_1, skip_tensor_2, cross_conv
