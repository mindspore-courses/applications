"""Base PIFuNet"""
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
from mindspore import nn

from .geometry import index, orthogonal, perspective


class BasePIFuNet(nn.Cell):
    """"Basic module of PIFuNet"""
    def __init__(
            self, projection_mode="orthogonal", error_term=nn.MSELoss(),
    ):
        super(BasePIFuNet, self).__init__()

        self.name = "base"

        self.error_term = error_term

        self.index = index

        self.projection = orthogonal if projection_mode == "orthogonal" else perspective

    def construct(self):
        """
        :param points: [B, 3, N] world space coordinates of points
        :param images: [B, C, H, W] input images
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :return: [B, Res, N] predictions for each point
        """

    def filter(self):
        """
        Filter the input images
        store all intermediate features.
        :param images: [B, C, H, W] input images
        """
        return None

    def query(self):
        """
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.
        :param points: [B, 3, N] world space coordinates of points
        :param calibs: [B, 3, 4] calibration matrices for each image
        :param transforms: Optional [B, 2, 3] image space coordinate transforms
        :param labels: Optional [B, Res, N] gt labeling
        :return: [B, Res, N] predictions for each point
        """
        return None

    def get_error(self, preds, labels):
        """
        Get the network loss from the last query
        :return: loss term
        """
        return self.error_term(preds, labels)
