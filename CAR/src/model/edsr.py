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
"""
Upsampling net. Employed to guide the training of the ResamplerNet.
"""

from mindspore import nn

from model.block import ResBlock, MeanShift, default_conv, Upsampler

class EDSR(nn.Cell):
    """
    Upscaling module to guide the training of the proposed CAR model.

    Args:
        n_resblocks(int): The number of net blocks. Default: 16.
        n_feats(int): The hidden layer features dimensions. default: 64.
        scale(int): Upscaling rate. Default: 4.
        conv(Cell): The convolution layer. Default: default_conv.

    Inputs:
        - **x** (Tensor) - The downscale image tensors.

    Outputs:
        Tensor, The super-resolution image tensors.
    """

    def __init__(self, n_resblocks=16, n_feats=64, scale=4, conv=default_conv):
        super(EDSR, self).__init__()

        kernel_size = 3
        act = nn.ReLU()
        self.sub_mean = MeanShift(1)
        self.add_mean = MeanShift(1, sign=1)

        # define head module
        m_head = [conv(3, n_feats, kernel_size)]

        # define body module
        m_body = [
            ResBlock(conv, n_feats, kernel_size, act=act, res_scale=0.1)
            for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, 3, kernel_size),
        ]

        self.head = nn.SequentialCell(m_head)
        self.body = nn.SequentialCell(*m_body)
        self.tail = nn.SequentialCell(*m_tail)

    def construct(self, x):
        """ build network """

        x = self.sub_mean(x)
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x
