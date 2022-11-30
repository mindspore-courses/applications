"""resnet module"""
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
import mindspore.nn as nn


class IUVPredictLayer(nn.Cell):
    """IUV predict"""
    def __init__(self, feat_dim=256, final_cov_k=3, with_uv=True):
        super().__init__()

        self.with_uv = with_uv

        if self.with_uv:
            self.predict_u = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=25,
                kernel_size=final_cov_k,
                stride=1,
                pad_mode="pad",
                padding=1 if final_cov_k == 3 else 0,
                has_bias=True,
            )

            self.predict_v = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=25,
                kernel_size=final_cov_k,
                stride=1,
                pad_mode="pad",
                padding=1 if final_cov_k == 3 else 0,
                has_bias=True,
            )

            self.predict_ann_index = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=15,
                kernel_size=final_cov_k,
                stride=1,
                pad_mode="pad",
                padding=1 if final_cov_k == 3 else 0,
                has_bias=True,
            )

            self.predict_uv_index = nn.Conv2d(
                in_channels=feat_dim,
                out_channels=25,
                kernel_size=final_cov_k,
                stride=1,
                pad_mode="pad",
                padding=1 if final_cov_k == 3 else 0,
                has_bias=True,
            )

            self.inplanes = feat_dim

    def _make_layer(self, block, planes, blocks, stride=1):
        """_make_layer"""
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.CellList(
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    has_bias=False,
                ),
                nn.BatchNorm2d(num_features=planes * block.expansion, momentum=0.9),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i < 0:
                continue
            layers.append(block(self.inplanes, planes))

        return nn.CellList(*layers)

    def forward(self, x_p):
        """forward"""
        return_dict = {}

        predict_uv_index = self.predict_uv_index(x_p)
        predict_ann_index = self.predict_ann_index(x_p)

        return_dict["predict_uv_index"] = predict_uv_index
        return_dict["predict_ann_index"] = predict_ann_index

        if self.with_uv:
            predict_u = self.predict_u(x_p)
            predict_v = self.predict_v(x_p)
            return_dict["predict_u"] = predict_u
            return_dict["predict_v"] = predict_v
        else:
            return_dict["predict_u"] = None
            return_dict["predict_v"] = None

        return return_dict
