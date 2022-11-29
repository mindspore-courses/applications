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
"""RetinaFace loss cell."""
from mindspore import nn


class RetinaFaceWithLossCell(nn.Cell):
    """
    RetinaFace with loss.

    Args:
        network(nn.Cell): The RetinaFace network structure.
        multibox_loss(nn.Cell): The MultiBoxLoss use for training.
        config(dict): A configuration dictionary, it should contains loc_weight, class_weight and landm_weight.

    Inputs:
        img(Tensor): Input images,which shape is [B,C,H,W].
        loc_t(Tensor): Ground truth of bounding boxes, load from dataset.
        conf_t(Tensor): Ground truth of confidence, load from dataset.
        landm_t(Tensor): Ground truth of landmarks, load from dataset.

    Returns:
        A tensor, represents the loss of forward pass.
    """

    def __init__(self, network, multibox_loss, config):
        super(RetinaFaceWithLossCell, self).__init__()
        self.network = network
        self.loc_weight = config['loc_weight']
        self.class_weight = config['class_weight']
        self.landm_weight = config['landm_weight']
        self.multibox_loss = multibox_loss

    def construct(self, img, loc_t, conf_t, landm_t):
        pred_loc, pre_conf, pre_landm = self.network(img)
        loss_loc, loss_conf, loss_landm = self.multibox_loss(pred_loc, loc_t, pre_conf, conf_t, pre_landm, landm_t)
        return loss_loc * self.loc_weight + loss_conf * self.class_weight + loss_landm * self.landm_weight
