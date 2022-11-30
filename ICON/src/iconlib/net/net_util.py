"""network utils"""
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
import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, Normal
from mindvision.classification.models.backbones.vgg import VGG19


def conv3x3(in_planes, out_planes, kernel=3, strd=1, dilation=1, padding=1, bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=kernel,
        dilation=dilation,
        stride=strd,
        pad_mode="pad",
        padding=padding,
        has_bias=bias,
    )


class ConvBlock(nn.Cell):
    """conv block"""
    def __init__(self, in_planes, out_planes, opt):
        super(ConvBlock, self).__init__()

        [k, s_p, d_p, p_p] = opt.conv3x3
        self.conv1 = conv3x3(in_planes, int(out_planes / 2), k, s_p, d_p, p_p)
        self.conv2 = conv3x3(int(out_planes / 2), int(out_planes / 4), k, s_p, d_p, p_p)
        self.conv3 = conv3x3(int(out_planes / 4), int(out_planes / 4), k, s_p, d_p, p_p)

        if opt.norm == "batch":
            self.bn1 = nn.BatchNorm2d(in_planes)
            self.bn2 = nn.BatchNorm2d(int(out_planes / 2))
            self.bn3 = nn.BatchNorm2d(int(out_planes / 4))
            self.bn4 = nn.BatchNorm2d(in_planes)
        elif opt.norm == "group":
            self.bn1 = nn.GroupNorm(32, in_planes)
            self.bn2 = nn.GroupNorm(32, int(out_planes / 2))
            self.bn3 = nn.GroupNorm(32, int(out_planes / 4))
            self.bn4 = nn.GroupNorm(32, in_planes)

        if in_planes != out_planes:
            self.downsample = nn.SequentialCell(
                [
                    nn.GroupNorm(32, in_planes),
                    nn.ReLU(),
                    nn.Conv2d(
                        in_planes, out_planes, kernel_size=1, stride=1, has_bias=False
                    ),
                ]
            )
        else:
            self.downsample = None

    def construct(self, x):
        """construct"""
        residual = x
        relu = ops.ReLU()

        out1 = self.bn1(x)
        out1 = relu(out1, True)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = relu(out2, True)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = relu(out3, True)
        out3 = self.conv3(out3)

        out3 = ops.concat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3


class Vgg19(nn.Cell):
    """Vgg19"""
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = VGG19(pretrained=True).features
        self.slice1 = nn.CellList()
        self.slice2 = nn.CellList()
        self.slice3 = nn.CellList()
        self.slice4 = nn.CellList()
        self.slice5 = nn.CellList()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x_p):
        """forward calculate"""
        h_relu1 = self.slice1(x_p)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class VGGLoss(nn.Cell):
    """Vgg loss"""
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):
        """forward calculate"""
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in enumerate(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


def init_net(net):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support);
    2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method:
        normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    # # apply 算子缺失，需要用nn.cells_and_names 来做
    # init_weights(net, init_type, init_gain=init_gain)
    return net


def init_weights(net, init_type="normal", init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        constant_init = mindspore.common.initializer.Constant(value=0.0)
        if hasattr(m, "weight") and (
                classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if init_type == "normal":
                m.weight.data = initializer(
                    Normal(mean=0.0, sigma=init_gain), m.weight.data
                )
            else:
                raise NotImplementedError(
                    f"initialization method [%s] is not implemented {init_type}"
                )
            if hasattr(m, "bias") and m.bias is not None:
                m.bias.data = constant_init(m.bias.data)
        elif (
                classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            m.weight.data = initializer(
                Normal(mean=1.0, sigma=init_gain), m.weight.data
            )
            m.bias.data = constant_init(m.bias.data)

    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>
