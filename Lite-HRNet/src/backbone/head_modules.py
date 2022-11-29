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

"""Head modules"""

import mindspore.nn as nn


class LiteTopDownSimpleHeatMap(nn.Cell):
    """
    Simple Implementation of TDSimpleHead in the original paper.

    Args:
        final_cfg (dict): Configs for the final layer.

            - num_channels of final_cfg (list): The number of input feature channel.
            - joints of final_cfg (int): The number of joints in predicted heatmaps.
            - final_conv_kernel (int): The kernel size of final_layer.

    Inputs:
        - **features** (Tensor) - Input feature tensor.

    Outputs:
        - **heatmap** (Tensor) - Predicted heatmap.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Example:
        >>> final_cfg = Dict("num_channels"=40, "joints"=17, "final_conv_kernel"=1)
        >>> heatmap_head = LiteTopDownSimpleHeatMap(final_cfg)
        >>> pred_heatmap = heatmap_head(mindspore.Tensor(np.random.rand(16, 40, 64, 48), mindspore.float32))
    """
    def __init__(self, final_cfg):
        super().__init__()
        self.final_layer = nn.Conv2d(
            in_channels=final_cfg["num_channels"],
            out_channels=final_cfg["joints"],
            kernel_size=final_cfg["final_conv_kernel"],
            stride=1,
            padding=1 if final_cfg["final_conv_kernel"] == 3 else 0,
            pad_mode="pad",
            has_bias=True)

    def construct(self, inputs):
        "Construct"

        pred_heatmap = self.final_layer(inputs)
        return pred_heatmap

class IterativeHead(nn.Cell):
    """
    Iterativehead that fuses the output from the stage 3.

    Args:
        in_channels (list): Input channel size.

    Inputs:
        - **features** (Tensor)- Input features from previous stages.

    Outputs:
        - **fused_features** (Tensor)- Fused output features.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Example:
        >>> in_channels = [40, 40, 40, 40]
        >>> iter_head = IterativeHead(in_channels)
        >>> fused_features = iter_head(mindspore.Tensor(np.random.rand(16, 4, 40, 64, 48), mindspore.float32))
    """
    def __init__(self, in_channels):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]

        for i in range(num_branchs):
            if i != num_branchs - 1:
                projects.append(
                    nn.SequentialCell(
                        [nn.Conv2d(self.in_channels[i],
                                   self.in_channels[i],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   group=self.in_channels[i],
                                   pad_mode="pad",
                                   has_bias=False),
                         nn.BatchNorm2d(self.in_channels[i]),
                         nn.Conv2d(self.in_channels[i],
                                   self.in_channels[i + 1],
                                   kernel_size=1,
                                   has_bias=False),
                         nn.BatchNorm2d(self.in_channels[i + 1]),
                         nn.ReLU()]
                    )
                )
            else:
                projects.append(
                    nn.SequentialCell(
                        [nn.Conv2d(self.in_channels[i],
                                   self.in_channels[i],
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   group=self.in_channels[i],
                                   pad_mode="pad",
                                   has_bias=False),
                         nn.BatchNorm2d(self.in_channels[i]),
                         nn.Conv2d(self.in_channels[i],
                                   self.in_channels[i],
                                   kernel_size=1,
                                   has_bias=False),
                         nn.BatchNorm2d(self.in_channels[i]),
                         nn.ReLU()]
                    )
                )
        self.projects = nn.CellList(projects)

    def construct(self, x):
        """Construct function."""

        x = x[::-1]
        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                resizer = nn.ResizeBilinear()
                last_x = resizer(
                    last_x,
                    size=s.shape[-2:],
                    align_corners=True)
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s

        return y[::-1]
