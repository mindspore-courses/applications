"""Normal map network"""
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
from mindspore import nn, ops, Tensor
import mindspore as ms

from iconlib.net.BasePIFuNet import BasePIFuNet
from iconlib.net.net_util import init_net, VGGLoss
from iconlib.net.FBNet import define_g


class NormalNet(BasePIFuNet):
    """
    HG PIFu network uses Hourglass stacks as the image filter.
    It does the following:
        1. Compute image feature stacks and store it in self.im_feat_list
            self.im_feat_list[-1] is the last stack (output stack)
        2. Calculate calibration
        3. If training, it index on every intermediate stacks,
            If testing, it index on the last stack.
        4. Classification.
        5. During training, error is calculated on all stacks.
    """

    def __init__(self, cfg, error_term=nn.SmoothL1Loss()):

        super(NormalNet, self).__init__(error_term=error_term)

        self.l1_loss = nn.SmoothL1Loss()

        self.opt = cfg.net

        if self.training:
            self.vgg_loss = [VGGLoss()]

        self.in_nml_f = [
            item[0] for item in self.opt.in_nml if "_F" in item[0] or item[0] == "image"
        ]
        self.in_nml_b = [
            item[0] for item in self.opt.in_nml if "_B" in item[0] or item[0] == "image"
        ]
        self.in_nml_f_dim = sum(
            [
                item[1]
                for item in self.opt.in_nml
                if "_F" in item[0] or item[0] == "image"
            ]
        )
        self.in_nml_b_dim = sum(
            [
                item[1]
                for item in self.opt.in_nml
                if "_B" in item[0] or item[0] == "image"
            ]
        )


        self.net_f = define_g(self.in_nml_f_dim, 3, 64, "global", 4, 9, "instance")
        self.net_b = define_g(self.in_nml_b_dim, 3, 64, "global", 4, 9, "instance")

        init_net(self)

    def construct(self, in_tensor):
        """construct"""
        inf_list = []
        inb_list = []

        for name in self.in_nml_f:
            inf_list.append(in_tensor[name])
        for name in self.in_nml_b:
            inb_list.append(in_tensor[name])

        nmlf = self.net_f(ops.concat(inf_list, axis=1))
        nmlb = self.net_b(ops.concat(inb_list, axis=1))
        # output: float_arr [-1,1] with [B, C, H, W]

        mask = ops.stop_gradient(Tensor((ops.abs(in_tensor['image'].sum(axis=1, keepdims=True)) !=
                                         0.0).asnumpy(), dtype=ms.float32))

        nmlf = nmlf * mask
        nmlb = nmlb * mask

        return nmlf, nmlb
