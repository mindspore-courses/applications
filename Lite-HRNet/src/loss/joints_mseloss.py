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

"""JointMSELoss"""

import mindspore.nn as nn
import mindspore.ops as ops


class JointsMSELoss(nn.Cell):
    """
    JointMSELoss

    Args:
        use_target_weight (bool): Whether using target weight

    Inputs:
        -**heatmap** (Tensor) - Output heatmap from Lite-HRNet.
        -**target_heatmap** (Tensor) - Target heatmap.
        -**weight** (Tensor) - Weights for each joint in the target.

    Outputs:
        -**loss_val** (Tensor) - Joints mse loss.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Example:
        >>> crit = JointMSELoss(False)
        >>> pred_heatmap = mindspore.Tensor(np.random.rand(4, 3, 256, 192), mindspore.float32)
        >>> target_heatmap = mindspore.Tensor(np.random.rand(4, 3, 256, 192), mindspore.float32)
        >>> loss_val = crit(pred_heatmap, target_heatmap)
    """
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def construct(self, output, target, weight):
        """Construct"""

        target_weight = weight
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        spliter = ops.Split(axis=1, output_num=num_joints)
        mul = ops.Mul()
        heatmaps_pred = spliter(output.reshape((batch_size, num_joints, -1)))
        heatmaps_gt = spliter(target.reshape((batch_size, num_joints, -1)))
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                heatmap_pred = mul(heatmap_pred, target_weight[:, idx])
                heatmap_gt = mul(heatmap_gt, target_weight[:, idx])
                loss += 0.5 * self.criterion(
                    heatmap_pred,
                    heatmap_gt
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints
