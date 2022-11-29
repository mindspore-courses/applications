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
"""JointsL2 Loss for MSPN"""
import mindspore
import mindspore.nn as nn


class StageLoss(nn.Cell):
    """ Stage Loss for Every Stage of MSPN

    Args:
        has_ohkm (bool): Whether to use Online Hard Key points Mining (OHKM). Default: False.
        topk (int): OHKM Top-k Largest Loss Hyper-parameter. Default: 8.
        vis_thresh_wo_ohkm (int): Joints Visible Thresh when has_ohkm sets to False. Default: 1.
        vis_thresh_w_ohkm (int): Joints Visible Thresh when has_ohkm sets to True. Default: 0.

    Inputs:
        - **output** (Tensor) - A tensor from MSPN output. The data type must be float32.
        - **valid** (Tensor) - A tensor of keypoints visible information, which has the same data type as `output`.
        - **label** (Tensor) - A tensor of keypoints ground truth, which has the same data type as `output`.

    Outputs:
        One Tensor, the MSPN Single-stage Model Loss.

        - **loss** (Tensor) - A tensor of MSPN Single-stage loss, has one dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> stage_loss = StageLoss(has_ohkm=True, topk=8)
    """
    def __init__(self,
                 has_ohkm: bool = False,
                 topk: int = 8,
                 vis_thresh_wo_ohkm: int = 1,
                 vis_thresh_w_ohkm: int = 0
                 ) -> None:
        super(StageLoss, self).__init__()
        self.has_ohkm = has_ohkm
        self.topk = topk
        self.t1 = vis_thresh_wo_ohkm
        self.t2 = vis_thresh_w_ohkm
        method = 'none' if self.has_ohkm else 'mean'
        self.calculate = nn.MSELoss(reduction=method)

    def construct(self, output, valid, label):
        """Construct Func"""
        greater = mindspore.ops.Greater()
        topk = mindspore.ops.TopK(sorted=False)
        batch_size = output.shape[0]
        keypoint_num = output.shape[1]
        loss = 0

        for i in range(batch_size):
            pred = output[i].reshape(keypoint_num, -1)
            gt = label[i].reshape(keypoint_num, -1)
            if not self.has_ohkm:
                weight = greater(valid[i], self.t1).astype(mindspore.float32)
                gt = gt * weight

            tmp_loss = self.calculate(pred, gt)
            if self.has_ohkm:
                tmp_loss = tmp_loss.mean(axis=1)
                weight = greater(valid[i].squeeze(), self.t2).astype(mindspore.float32)
                tmp_loss = tmp_loss * weight
                topk_val, _ = topk(tmp_loss, self.topk)
                sample_loss = topk_val.mean(axis=0)
            else:
                sample_loss = tmp_loss

            loss = loss + sample_loss

        return loss / batch_size


class JointsL2Loss(nn.Cell):
    """ JointsL2 Loss for MSPN

    Args:
        stage_num (int): MSPN Stage Num.
        ctf (bool): Whether to enable Coarse-to-Fine Supervision. Default: True.
        has_ohkm (bool): Whether to use Online Hard Key points Mining (OHKM). Default: False.
        topk (int): OHKM Top-k Largest Loss Hyper-parameter. Default: 8.

    Inputs:
        - **output** (Tensor) - A tensor from MSPN output. The data type must be float32.
        - **valid** (Tensor) - A tensor of keypoints visible information, which has the same data type as `output`.
        - **label** (Tensor) - A tensor of keypoints ground truth, which has the same data type as `output`.

    Outputs:
        One Tensor, the MSPN Multi-stage Model Loss.

        - **loss** (Tensor) - A tensor of MSPN Multi-stage loss, has one dimension.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> joints_loss = JointsL2Loss(stage_num=0, ctf=True, has_ohkm=True, topk=8)
    """
    def __init__(self,
                 stage_num: int,
                 ctf: bool = True,
                 has_ohkm: bool = False,
                 topk: int = 8
                 ) -> None:
        super(JointsL2Loss, self).__init__()
        self.stage_num = stage_num
        self.ctf = ctf
        self.ohkm = has_ohkm
        self.topk = topk
        self.loss = StageLoss()
        self.loss_ohkm = StageLoss(has_ohkm=self.ohkm, topk=self.topk)

    def construct(self, outputs, valids, labels):
        """Construct Func"""
        loss = 0
        for i in range(self.stage_num):
            for j in range(4):
                ind = j
                if i == self.stage_num - 1 and self.ctf:
                    ind += 1
                tmp_labels = labels[:, ind, :, :, :]

                if j == 3 and self.ohkm:
                    tmp_loss = self.loss_ohkm(outputs[i][j], valids, tmp_labels)
                else:
                    tmp_loss = self.loss(outputs[i][j], valids, tmp_labels)

                if j < 3:
                    tmp_loss = tmp_loss / 4

                loss += tmp_loss

        return loss
