"""MLP module"""
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
from mindspore import nn, ops


class MLP(nn.Cell):
    """MLP module"""
    def __init__(
            self, filter_channels, name=None, res_layers=None, norm="group", last_op=None
    ):
        if res_layers is None:
            res_layers = []
        super(MLP, self).__init__()

        filter_layers = []
        norms_layers = []
        self.res_layers = res_layers
        self.norm = norm
        self.last_op = last_op
        self.name = name
        self.activate = nn.LeakyReLU()

        for l in range(0, len(filter_channels) - 1):
            if l in self.res_layers:
                filter_layers.append(
                    nn.Conv1d(
                        filter_channels[l] + filter_channels[0],
                        filter_channels[l + 1],
                        1,
                        has_bias=True,
                    )
                )
            else:
                filter_layers.append(
                    nn.Conv1d(
                        filter_channels[l], filter_channels[l + 1], 1, has_bias=True
                    )
                )

            if l != len(filter_channels) - 2:
                if norm == "group":
                    norms_layers.append(nn.GroupNorm(32, filter_channels[l + 1]))
                elif norm == "batch":
                    norms_layers.append(nn.BatchNorm1d(filter_channels[l + 1]))
                elif norm == "instance":
                    norms_layers.append(nn.InstanceNorm1d(filter_channels[l + 1]))
        self.filters = nn.CellList(filter_layers)
        self.norms = nn.CellList(norms_layers)

    def construct(self, feature):
        """
        feature may include multiple view inputs
        args:
            feature: [B, C_in, N]
        return:
            [B, C_out, N] prediction
        """
        y = feature
        tmpy = feature

        for i, f in enumerate(self.filters):

            y = f(y if i not in self.res_layers else ops.concat([y, tmpy], 1))
            if i != len(self.filters) - 1:
                if self.norm not in ["batch", "group", "instance"]:
                    y = self.activate(y)
                else:
                    y = self.activate(self.norms[i](y))

        if self.last_op is not None:
            y = self.last_op(y)

        return y
