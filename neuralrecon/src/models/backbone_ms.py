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
"""Backbone"""

import time

import numpy as np
import mindspore as ms
from mindspore import ops, nn
import mindspore_hub as mshub


def _round_to_multiple_of(val, divisor, round_up_has_bias=0.9):
    """ Asymmetric rounding to make `val` divisible by `divisor`. With default
    has_bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    assert 0.0 < round_up_has_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)
    return new_val if new_val >= round_up_has_bias * val else new_val + divisor


def _get_depths(alpha):
    """ Scales tensor depths as in reference MobileNet code, prefers rounding up
    rather than down. """
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MnasMulti(nn.Cell):
    """Mnasnet backbone"""

    def __init__(self, alpha=1.0):
        """Init"""
        super(MnasMulti, self).__init__()
        depths = _get_depths(alpha)
        model_name = 'mindspore/1.8/mnasnet_imagenet2012'
        mnasnet = mshub.load(model_name, pretrained=True, force_reload=False)

        self.conv0 = nn.SequentialCell(
            mnasnet.features._cells['0'],   # pylint: disable-protected-access
            mnasnet.features._cells['1'],   # pylint: disable-protected-access
            mnasnet.features._cells['2'],   # pylint: disable-protected-access
            mnasnet.features._cells['3'],   # pylint: disable-protected-access
            mnasnet.features._cells['4']    # pylint: disable-protected-access
        )
        # To match PyTorch version of MnasMulti
        self.conv0[0].conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3),
                                       stride=(2, 2), pad_mode='pad', padding=1, has_bias=False)
        self.conv0[1].sequence[3] = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(1, 1),
                                              stride=(1, 1), has_bias=False)

        self.conv1 = nn.SequentialCell(
            mnasnet.features._cells['5'],   # pylint: disable-protected-access
            mnasnet.features._cells['6'],   # pylint: disable-protected-access
            mnasnet.features._cells['7'],   # pylint: disable-protected-access
        )
        self.conv2 = nn.SequentialCell(
            mnasnet.features._cells['8'],   # pylint: disable-protected-access
            mnasnet.features._cells['9'],   # pylint: disable-protected-access
            mnasnet.features._cells['10'],  # pylint: disable-protected-access
        )

        self.out1 = nn.Conv2d(depths[4], depths[4], 1, has_bias=False)
        self.out_channels = [depths[4]]

        final_chs = depths[4]
        self.inner1 = nn.Conv2d(depths[3], final_chs, 1, has_bias=True)
        self.inner2 = nn.Conv2d(depths[2], final_chs, 1, has_bias=True)

        self.out2 = nn.Conv2d(final_chs, depths[3], 3, pad_mode="pad", padding=1, has_bias=False)
        self.out3 = nn.Conv2d(final_chs, depths[2], 3, pad_mode="pad", padding=1, has_bias=False)
        self.out_channels.append(depths[3])
        self.out_channels.append(depths[2])

    def construct(self, x):
        """Construct"""
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = []
        out = self.out1(intra_feat)
        outputs.append(out)

        h, w = intra_feat.shape[2:]
        intra_feat = ops.ResizeNearestNeighbor((h * 2, w * 2))(intra_feat) + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs.append(out)

        h, w = intra_feat.shape[2:]
        intra_feat = ops.ResizeNearestNeighbor((h * 2, w * 2))(intra_feat) + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs.append(out)

        return outputs[::-1]


if __name__ == '__main__':
    # ms.set_context(mode=ms.PYNATIVE_MODE, device_target='GPU')
    ms.set_context(mode=ms.GRAPH_MODE, device_target='GPU')
    model = MnasMulti()

    input_tensor = ms.Tensor(np.random.randint(0, 10, size=(1, 3, 640, 480)).astype(np.float32))
    start_time = time.time()
    for i in range(1000):
        output = model(input_tensor)
    print('inference fps:', 1000 / (time.time() - start_time))
