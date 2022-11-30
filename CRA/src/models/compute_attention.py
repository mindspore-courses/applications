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
"""Attention computing and transfer module."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops

from .network_module import GatedConv2d


def downsample(x):
    """
    Downsample the input image to half the original size.

    Args:
        x(Tensor): input to be sampled.

    Return:
        x: Output after downsampling.
    """

    shp = x.shape
    net = nn.Unfold([1, 1, 1, 1], [1, 2, 2, 1], [1, 1, 1, 1], 'same')
    x = net(x)
    x = ops.Reshape()(x, (shp[0], shp[1], shp[2] // 2, shp[3] // 2))
    return x


class ContextualAttention(nn.Cell):
    """
    Attention score computing module.

    Args:
        softmax_scale(int): scaled softmax for attention.
        src(Tensor): input feature to match (foreground).
        ref(Tensor): input feature for match (background).
        mask(Tensor): input mask for ref, indicating patches not available.

    Return:
        out: Foreground area filled with context information
             (It generally refers to the 64 * 64 feature map used to calculate attention scores).
        correspondence: Attention score.
    """

    def __init__(self, softmax_scale=10, fuse=True, dtype=mindspore.float32):
        super(ContextualAttention, self).__init__()
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.dtype = dtype
        self.reducesum = ops.ReduceSum(False)
        self.unfold1 = nn.Unfold([1, 3, 3, 1], [1, 2, 2, 1], [1, 1, 1, 1], 'same')
        self.unfold2 = nn.Unfold([1, 3, 3, 1], [1, 1, 1, 1], [1, 1, 1, 1], 'same')
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.pool1 = nn.MaxPool2d(16, 16, 'same', 'NCHW')
        self.pool2 = nn.MaxPool2d(3, 1, 'same', 'NCHW')
        self.maximum = ops.Maximum()
        self.sqrt = ops.Sqrt()
        self.square = ops.Square()
        self.eye = ops.Eye()
        self.reducemax = ops.ReduceMax(True)
        self.greaterequal = ops.GreaterEqual()
        self.pow = ops.Pow()
        self.div = ops.Div()
        self.softmax = nn.Softmax(1)
        self.cat = ops.Concat(0)
        self.conv1 = InitConv2d([3, 3, 128, 1024], 1, True)
        self.conv2 = InitConv2d([3, 3, 1, 1], 1, True)
        self.disconv1 = InitConv2d([3, 3, 128, 1024], 2, False)

    def construct(self, src, ref, mask, method='SOFT'):
        """compute attention score"""

        # get shapes
        shape_src = src.shape
        batch_size = shape_src[0]
        nc = shape_src[1]
        # raw features
        raw_feats = self.unfold1(ref)
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 1))
        raw_feats = self.reshape(raw_feats, (batch_size, -1, 3, 3, nc))
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 4, 1))
        split = ops.Split(0, batch_size)
        raw_feats_lst = split(raw_feats)
        # resize
        src = downsample(src)
        ref = downsample(ref)
        ss = src.shape
        rs = ref.shape
        src_lst = split(src)
        feats = self.unfold2(ref)
        feats = self.transpose(feats, (0, 2, 3, 1))
        feats = self.reshape(feats, (batch_size, -1, 3, 3, nc))
        feats = self.transpose(feats, (0, 2, 3, 4, 1))
        feats_lst = split(feats)
        # process mask
        mask = self.pool1(mask)
        mask = self.pool2(mask)
        mask = 1 - mask
        mask = self.reshape(mask, (1, -1, 1, 1))

        y_lst, y_up_lst = [], []
        offsets = []
        fuse_weight = self.reshape(self.eye(3, 3, mindspore.float32), (3, 3, 1, 1))
        for x, r, raw_r in zip(src_lst, feats_lst, raw_feats_lst):
            r = r[0]
            r = r / self.maximum(self.sqrt(self.reducesum(self.square(r), [0, 1, 2])), 1e-8)
            r_kernel = self.transpose(r, (3, 2, 0, 1))
            y = self.conv1(x, r_kernel)
            if self.fuse:
                # conv implementation for fuse scores to encourage large patches
                yi = self.reshape(y, (1, 1, ss[2] * ss[3], rs[2] * rs[3]))
                fuse_weight_kernel = ops.Transpose()(fuse_weight, (3, 2, 0, 1))
                yi = self.conv2(yi, fuse_weight_kernel)
                yi = self.transpose(yi, (0, 2, 3, 1))
                yi = self.reshape(yi, (1, ss[2], ss[3], rs[2], rs[3]))
                yi = self.transpose(yi, (0, 2, 1, 4, 3))
                yi = self.reshape(yi, (1, ss[2] * ss[3], rs[2] * rs[3], 1))
                yi = self.transpose(yi, (0, 3, 1, 2))
                yi = self.conv2(yi, fuse_weight_kernel)
                yi = self.transpose(yi, (0, 2, 3, 1))
                yi = self.reshape(yi, (1, ss[3], ss[2], rs[3], rs[2]))
                yi = self.transpose(yi, (0, 2, 1, 4, 3))
                y = yi
            y = self.reshape(y, (1, ss[2], ss[3], rs[2] * rs[3]))
            y = self.transpose(y, (0, 3, 1, 2))
            if method == 'HARD':
                ym = self.reducemax(y, 1)
                y = y * mask
                coef = self.greaterequal(y, max(y, 1)).astype(self.dtype)
                y = self.pow(coef * self.div(y, ym + 1e-04), 2)
            elif method == 'SOFT':
                y = (self.softmax(y * mask * self.softmax_scale)) * mask
            y = self.reshape(y, (1, rs[2] * rs[3], ss[2], ss[3]))
            if self.dtype == mindspore.float32:
                offset = y.argmax(1)
                offsets.append(offset)
            feats = raw_r[0]
            feats_kernel = self.transpose(feats, (3, 2, 0, 1))
            y_up = self.disconv1(y, feats_kernel)
            y_lst.append(y)
            y_up_lst.append(y_up)
        out, correspondence = self.cat(y_up_lst), self.cat(y_lst)
        out = self.reshape(out, (shape_src[0], shape_src[1], shape_src[2], shape_src[3]))
        return out, correspondence


def freeze(layer):
    """
    Freeze network layer parameters to prevent reverse updates.

    Args:
        layer(cell): the network layer to be freezed.
    """

    for param in layer.get_parameters():
        param.requires_grad = False


class InitConv2d(nn.Cell):
    """
    Assign a fixed value(w) to the convolution layer weight parameter.

    Args:
        shape(list): specify input and output channels and convolution kernel size of convolutional network.
        rate(int): the moving step of convolution kernel.
        con_dis(bool): If true,it is Conv2d; Otherwise,it is Conv2dTranspose.

    Return:
        out: Convolution network layer output result with given weight value.
    """

    def __init__(self, shape, rate=1, con_dis=True):
        super(InitConv2d, self).__init__()
        self.shape = shape
        self.rate = rate
        self.con_dis = con_dis
        self.h, self.w, self.i, self.o = self.shape[0], self.shape[1], self.shape[2], self.shape[3]
        if self.con_dis:
            self.conv = nn.Conv2d(self.i, self.o, (self.h, self.w), (1, 1), 'same')
            freeze(self.conv)
        else:
            self.conv = nn.Conv2dTranspose(self.o, self.i, (self.h, self.w), (self.rate, self.rate), 'same')
            freeze(self.conv)
        self.tmp = mindspore.ParameterTuple(self.get_parameters())

    def construct(self, x, w):
        for weight in self.tmp:
            ops.Assign()(weight, w)
        out = self.conv(x)
        return out


class ApplyAttention(nn.Cell):
    """
    Attention transfer module(used for training)
    (It generally used for 128 * 128 / 256 * 256 feature map).

    Args:
        shp(list): the shape of input feature map.
        shp_att(list): the shape of attention score.

    Return:
        out: Feature map filled by attention transfer module.
    """

    def __init__(self, shp, shp_att):
        super(ApplyAttention, self).__init__()
        self.shp = shp
        self.shp_att = shp_att
        self.rate = self.shp[2] // self.shp_att[2]
        self.kernel = self.rate * 2
        self.batch_size = self.shp[0]
        self.sz = self.shp[2]
        self.nc = self.shp[1]
        self.unfold = nn.Unfold([1, self.kernel, self.kernel, 1], [1, self.rate, self.rate, 1], [1, 1, 1, 1], 'same')
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.split = ops.Split(0, self.batch_size)
        self.disconv1 = InitConv2d([8, 8, 64, 1024], self.rate, False)
        self.disconv2 = InitConv2d([16, 16, 32, 1024], self.rate, False)
        self.concat = ops.Concat(0)
        self.conv_pl2 = nn.SequentialCell(
            GatedConv2d(64, 64, 3, 1, 1),
            GatedConv2d(64, 64, 3, 1, 2)
        )
        self.conv_pl1 = nn.SequentialCell(
            GatedConv2d(32, 32, 3, 1, 1),
            GatedConv2d(32, 32, 3, 1, 2)
        )

    def construct(self, x, correspondence):
        """apply attention on training"""

        raw_feats = self.unfold(x)
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 1))
        raw_feats = self.reshape(raw_feats, (self.batch_size, -1, self.kernel, self.kernel, self.nc))
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 4, 1))
        raw_feats_lst = self.split(raw_feats)
        ys = []
        correspondence = self.transpose(correspondence, (0, 2, 3, 1))
        att_lst = self.split(correspondence)
        for feats, att in zip(raw_feats_lst, att_lst):
            feats_kernel = self.transpose(feats[0], (3, 2, 0, 1))
            att = self.transpose(att, (0, 3, 1, 2))
            if self.shp[2] == 128:
                y1 = self.disconv1(att, feats_kernel)
                ys.append(y1)
            elif self.shp[2] == 256:
                y2 = self.disconv2(att, feats_kernel)
                ys.append(y2)
            else:
                print('Value Error')
        out = self.concat(ys)
        if self.shp[2] == 128:
            out = self.conv_pl2(out)
        elif self.shp[2] == 256:
            out = self.conv_pl1(out)
        else:
            print('conv error')
        return out


class ApplyAttention2(nn.Cell):
    """
    Attention transfer module(used for testing post-processing part).
    Apply attention transfer module to high-frequency residual image.

    Args:
        shp(list): the shape of input residual image.
        shp_att(list): the shape of attention score.

    Return:
        out: Aggregated residual image filled by attention transfer module.
    """

    def __init__(self, shp, shp_att):
        super(ApplyAttention2, self).__init__()
        self.shp = shp
        self.shp_att = shp_att
        self.rate = self.shp[2] // self.shp_att[2]
        self.kernel = self.rate
        self.batch_size = self.shp[0]
        self.sz = self.shp[2]
        self.nc = self.shp[1]
        self.unfold = nn.Unfold([1, self.kernel, self.kernel, 1], [1, self.rate, self.rate, 1], [1, 1, 1, 1], 'same')
        self.transpose = ops.Transpose()
        self.reshape = ops.Reshape()
        self.split = ops.Split(0, self.batch_size)
        self.disconv1 = InitConv2d([128, 128, 3, 1024], self.rate, False)
        self.concat = ops.Concat(0)

    def construct(self, x, correspondence):
        """apply attention on testing"""

        raw_feats = self.unfold(x)
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 1))
        raw_feats = self.reshape(raw_feats, (self.batch_size, -1, self.kernel, self.kernel, self.nc))
        raw_feats = self.transpose(raw_feats, (0, 2, 3, 4, 1))
        raw_feats_lst = self.split(raw_feats)
        ys = []
        correspondence = self.transpose(correspondence, (0, 2, 3, 1))
        att_lst = self.split(correspondence)
        for feats, att in zip(raw_feats_lst, att_lst):
            feats_kernel = self.transpose(feats[0], (3, 2, 0, 1))
            att = self.transpose(att, (0, 3, 1, 2))
            y = self.disconv1(att, feats_kernel)
            ys.append(y)
        out = self.concat(ys)
        return out
