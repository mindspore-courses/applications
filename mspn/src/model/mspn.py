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
"""MSPN Model"""
import mindspore.nn as nn

from src.model.blocks import ResNetTop
from src.model.mspn_single_stage import SingleStageModule
from src.utils.jointsl2loss import JointsL2Loss


class MSPN(nn.Cell):
    """ MSPN

    Args:
        total_stage_num (int): MSPN Stage Num.
        output_channel_num (int): Output Tensor Channels.
        output_shape (tuple): Output Tensor Shape.
        upsample_channel_num (int): Upsample Tensor Channels.
        online_hard_key_mining (bool): Whether to use Online Hard Key points Mining (OHKM). Default: True.
        topk_keys (int): OHKM Top-k Largest Loss Hyper-parameter. Default: 8.
        coarse_to_fine (bool): Whether to enable Coarse-to-Fine Supervision. Default: True.

    Inputs:
        - **imgs** (Tensor) - An image tensor of MSPN input. The data type must be float32 and has four dimensions.
        - **valids** (Tensor) - A tensor of keypoints visible information, which has the same data type as `imgs`.
        - **labels** (Tensor) - A tensor of keypoints ground truth, which has the same data type as `imgs`.

    Outputs:
        One Tensor, the MSPN Multi-stage Model Loss when training otherwise the predicted human keypoints of input data.

        - **output** (Tensor) - A tensor of MSPN Multi-stage loss when training otherwise the predicted human \
        keypoints of input data.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> mspn = MSPN(2, 17, (64, 48), 256, ohkm=True, topk=8, ctf=True)
    """
    def __init__(self,
                 total_stage_num: int,
                 output_channel_num: int,
                 output_shape: tuple,
                 upsample_channel_num: int,
                 online_hard_key_mining: bool = True,
                 topk_keys: int = 8,
                 coarse_to_fine: bool = True,
                 ) -> None:
        super(MSPN, self).__init__()
        self.top = ResNetTop()
        self.total_stage_num = total_stage_num
        self.output_channel_num = output_channel_num
        self.output_shape = output_shape
        self.upsample_channel_num = upsample_channel_num
        self.online_hard_key_mining = online_hard_key_mining
        self.topk_keys = topk_keys
        self.coarse_to_fine = coarse_to_fine
        self.mspn_modules = list()
        for i in range(self.total_stage_num):
            if i == 0:
                use_skip = False
            else:
                use_skip = True
            if i != self.total_stage_num - 1:
                generate_skip = True
                generate_cross_conv = True
            else:
                generate_skip = False
                generate_cross_conv = False
            self.mspn_modules.append(
                SingleStageModule(
                    self.output_channel_num, self.output_shape,
                    use_skip=use_skip, generate_skip=generate_skip,
                    generate_cross_conv=generate_cross_conv,
                    channel_num=self.upsample_channel_num,
                )
            )
            setattr(self, 'stage%d' % i, self.mspn_modules[i])
        self.loss = JointsL2Loss(stage_num=self.total_stage_num, ctf=self.coarse_to_fine,
                                 has_ohkm=self.online_hard_key_mining, topk=self.topk_keys)

    def construct(self, imgs, valids=None, labels=None):
        """Construct Func"""
        x = self.top(imgs)
        skip_tensor_1 = None
        skip_tensor_2 = None
        outputs = list()
        for i in range(self.total_stage_num):
            res, skip_tensor_1, skip_tensor_2, x = self.mspn_modules[i](x, skip_tensor_1, skip_tensor_2)
            outputs.append(res)

        if valids is None and labels is None:
            return outputs[-1][-1]

        return self.loss(outputs, valids, labels)
