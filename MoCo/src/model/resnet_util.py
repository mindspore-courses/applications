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
"""resnet18 for the MoCo"""

import math
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import constexpr

from moco_utils.args import parse_args

args = parse_args()


class BasicBlock(nn.Cell):
    """
    BasicBlock of ResNet18

    Args:
        in_planes: Input channel
        planes:  Output channel
        kernel_size: Convolution kernel size
        stride: Convolution step
    """
    def __init__(self, in_planes, planes, kernel_size=3, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, padding=1, has_bias=False,
                               pad_mode='pad', bias_init="zeros", data_format="NCHW")
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.99, affine=True, gamma_init='ones',
                                  beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones',
                                  use_batch_statistics=None, data_format='NCHW')
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=1, padding=1, has_bias=False,
                               pad_mode='pad', bias_init="zeros", data_format="NCHW")
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5, momentum=0.99, affine=True, gamma_init='ones',
                                  beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones',
                                  use_batch_statistics=None, data_format='NCHW')
        if stride != 1 or in_planes != planes:
            self.downsample = nn.SequentialCell(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, has_bias=True, pad_mode='valid',
                          bias_init="zeros", data_format="NCHW"),
                nn.BatchNorm2d(planes, eps=1e-5, momentum=0.99, affine=True, gamma_init='ones',
                               beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones',
                               use_batch_statistics=None, data_format='NCHW'))
        else:
            self.downsample = nn.SequentialCell()

    def construct(self, inx):
        """calculate basic module output."""
        x = self.relu(self.bn1(self.conv1(inx)))
        x = self.bn2(self.conv2(x))
        out = x + self.downsample(inx)
        out = self.relu(out)
        return out


@constexpr
def compute_kernel_size(inp_shape, output_size):
    """AdaptiveAvgPool2d script."""
    kernel_width, kernel_height = inp_shape[2], inp_shape[3]
    if isinstance(output_size, int):
        kernel_width = math.ceil(kernel_width / output_size)
        kernel_height = math.ceil(kernel_height / output_size)
    elif isinstance(output_size, (list, tuple)):
        kernel_width = math.ceil(kernel_width / output_size[0])
        kernel_height = math.ceil(kernel_height / output_size[1])

    return kernel_width, kernel_height


class AdaptiveAvgPool2d(nn.Cell):
    """build AdaptiveAvgPool2d for Ascend."""
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size

    def construct(self, x):
        inp_shape = x.shape
        kernel_size = compute_kernel_size(inp_shape, self.output_size)
        return ops.AvgPool(kernel_size, kernel_size)(x)


class ResNet18(nn.Cell):
    """backbone of MoCo"""
    def __init__(self, basicblocks, blocknums, nb_classes):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=3, stride=1, padding=1, has_bias=False,
                               pad_mode='pad', bias_init="zeros", data_format="NCHW")
        self.bn1 = nn.BatchNorm2d(self.in_planes, eps=1e-5, momentum=0.99, affine=True, gamma_init='ones',
                                  beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones',
                                  use_batch_statistics=None, data_format='NCHW')
        self.relu = nn.ReLU()

        self.layer1 = self._make_layers(basicblocks, blocknums[0], 64, 1)
        self.layer2 = self._make_layers(basicblocks, blocknums[1], 128, 2)
        self.layer3 = self._make_layers(basicblocks, blocknums[2], 256, 2)
        self.layer4 = self._make_layers(basicblocks, blocknums[3], 512, 2)

        if args.device_target == 'Ascend':
            self.avgpool = AdaptiveAvgPool2d((1, 1))
        else:
            self.avgpool = ops.AdaptiveAvgPool2D((1, 1))

        self.flatten = nn.Flatten()
        self.fc = nn.Dense(in_channels=512, out_channels=nb_classes,
                           weight_init='normal', bias_init='zeros', has_bias=True)

    def _make_layers(self, basicblock, blocknum, plane, stride):
        """
        make_layers for ResNet18

        Args:
            basicblock: Basic residual block class
            blocknum: The number of basic residual blocks in the current layer is 2 for each layer of resnet18
            plane: Number of output channels
            stride: Convolution step
        """
        layers = []
        for i in range(blocknum):
            if i == 0:
                layer = basicblock(self.in_planes, plane, 3, stride=stride)
            else:
                layer = basicblock(plane, plane, 3, stride=1)
            layers.append(layer)
        self.in_planes = plane
        return nn.SequentialCell(*layers)

    def construct(self, inx):
        """calculate ResNet18 output."""
        x = self.relu(self.bn1(self.conv1(inx)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        out = self.fc(x)

        return out
