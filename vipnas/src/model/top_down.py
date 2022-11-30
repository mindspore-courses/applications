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
"""Define Top-down model"""

import numpy as np
import mindspore.nn as nn
from mindspore import Tensor
from vipnas_heatmap_simple_head import ViPNASHeatmapSimpleHead
from vipnas_mbv3 import ViPNASMobileNetV3
from vipnas_resnet import ViPNASResNet


class TopDown(nn.Cell):
    """Top-down pose detectors.

    Args:
        backbone (nn.Cell): Backbone to generate feature.
        train_cfg (dict): Config for training. Default: None.
        test_cfg (dict): Config for testing. Default: None.
    """

    def __init__(self,
                 backbone=None,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()

        if test_cfg is None:
            test_cfg = dict(
                flip_test=True,
                shift_heatmap=True)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        if backbone == 'ViPNAS_ResNet':
            self.backbone = ViPNASResNet(depth=50)
            self.keypoint_head = ViPNASHeatmapSimpleHead(
                in_channels=608,
                out_channels=17,
                train_cfg=self.train_cfg,
                test_cfg=self.test_cfg
            )
        elif backbone == 'ViPNAS_MobileNetV3':
            self.backbone = ViPNASMobileNetV3()
            self.keypoint_head = ViPNASHeatmapSimpleHead(
                in_channels=160,
                out_channels=17,
                num_deconv_filters=(160, 160, 160),
                num_deconv_groups=(160, 160, 160),
                train_cfg=self.train_cfg,
                test_cfg=self.test_cfg
            )

    def construct(self,
                  img,
                  img_metas=None,
                  return_loss=True,
                  return_heatmap=False):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.

        Note:
            batch_size: N
            num_keypoints: K
            num_img_channel: C (Default: 3)
            img height: imgH
            img width: imgW
            heatmaps height: H
            heatmaps weight: W

        Args:
            img (mindspore.Tensor[NxCximgHximgW]): Input images.
            img_metas (list(dict)): Information about data augmentation
                By default this includes:
                - "image_file: path to the image file
                - "center": center of the bbox
                - "scale": scale of the bbox
                - "rotation": rotation of the bbox
                - "bbox_score": score of bbox
            return_loss (bool): Option to `return loss`. `return loss=True`
                for training, `return loss=False` for validation & test.
            return_heatmap (bool) : Option to return heatmap.

        Returns:
            dict|tuple: if `return loss` is true, then return losses.
              Otherwise, return predicted poses, boxes, image paths
                  and heatmaps.
        """
        if return_loss:
            return self.forward_train(img)
        return self.forward_test(
            img, img_metas, return_heatmap=return_heatmap)

    def forward_train(self, img):
        """Defines the computation performed at every call when training."""
        output = self.backbone.construct(img)
        output = self.keypoint_head.construct(output)

        return output

    def forward_test(self, img, img_metas, return_heatmap=False):
        """Defines the computation performed at every call when testing."""
        if img.shape[0] != len(img_metas):
            raise ValueError('img size dose not match img_metas')
        batch_size, _, _, _ = img.shape
        if batch_size > 1:
            if 'bbox_id' not in img_metas[0]:
                raise ValueError('bbox_id not in img_metas[0]')

        result = {}

        features = self.backbone.construct(img)
        output_heatmap = self.keypoint_head.inference_model(
            features, flip_pairs=None)

        if self.test_cfg.get('flip_test', True):
            img_flipped = Tensor(np.flip(img.asnumpy(), axis=3))
            features_flipped = self.backbone.construct(img_flipped)
            output_flipped_heatmap = self.keypoint_head.inference_model(
                features_flipped, img_metas[0]['flip_pairs'])
            output_heatmap = (output_heatmap +
                              output_flipped_heatmap) * 0.5

        keypoint_result = self.keypoint_head.decode(
            img_metas, output_heatmap)
        result.update(keypoint_result)

        if not return_heatmap:
            output_heatmap = None

            result['output_heatmap'] = output_heatmap

        return result


def create_net(backbone):
    network = TopDown(backbone=backbone,
                      test_cfg=dict(
                          flip_test=True,
                          shift_heatmap=True))

    return network
