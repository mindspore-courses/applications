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

"""Mindspore implementation of Lite-HRNet"""

from mindspore import nn
from mindspore.nn.layer.normalization import _BatchNorm

from backbone.ccw_modules import ConditionalChannelWeighting
from backbone.funcs import IdentityMap, UpsampleNearest
from backbone.head_modules import IterativeHead, LiteTopDownSimpleHeatMap
from backbone.shuffle_unit import ShuffleUnit
from backbone.stem import Stem


class LiteHRModule(nn.Cell):
    """
    Definition of the module in a stage

    Args:
        num_branches (int): The number of branches in this module.
        num_blocks (int): The number of blocks.
        in_channels (int): Input channel size.
        reduce_ratio (int): The ratio of input channel size to intermediate channel size.
        module_type (str): the type of this module, Naive or Lite.
        multiscale_output (bool): Whether the output have different resolution. Default: False.
        with_fuse (bool): Whether this module have fusing layers at the end. Default: True.

    Inputs:
        - **features** (Tensor) - Input image tensors from dataset API.

    Outputs:
        - **output_features** (Tensor) - Predicted joint heatmaps.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Example:
        >>> shuffle_unit = ShuffleUnit(3, 3)
        >>> output_features = shuffle_unit(mindspore.Tensor(np.random.rand(16, 3, 64, 48), mindspore.float32))
    """

    def __init__(
            self,
            num_branches,
            num_blocks,
            in_channels,
            reduce_ratio,
            module_type,
            multiscale_output=False,
            with_fuse=True
    ):
        super().__init__()
        self._check_branches(num_branches, in_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches

        self.module_type = module_type
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse

        if self.module_type == 'LITE':
            self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
        elif self.module_type == 'NAIVE':
            self.layers = self._make_naive_branches(num_branches, num_blocks)
        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        """Check inputs to avoid ValueError"""

        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        """
        Make conditional channel weighting blocks.

        Args:
            num_blocks (int): The number of blocks in this layer.
            reduce_ratio (int): The ratio of input channel size to intermediate channel size.
            stride (int): Stride. Default: 1.
        """

        layers = nn.SequentialCell()
        for i in range(num_blocks):
            i = i * 1
            layers.append(
                ConditionalChannelWeighting(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio))

        return layers

    def _make_one_branch(self, branch_index, num_blocks, stride=1):
        """
        Make one naive branch.

        Args:
            branch_index (int): Current branch index.
            num_blocks(int): The number of blocks in this branch.
            stride (int): Stride. Default: 1.
        """

        layers = []
        layers.append(
            ShuffleUnit(
                self.in_channels[branch_index],
                self.in_channels[branch_index],
                stride=stride))
        for i in range(1, num_blocks):
            i = i * 1
            layers.append(
                ShuffleUnit(
                    self.in_channels[branch_index],
                    self.in_channels[branch_index],
                    stride=1))

        return nn.SequentialCell(layers)

    def _make_naive_branches(self, num_branches, num_blocks):
        """
        Make branches.

        Args:
            num_branches (int): The number of resolution branches.
            num_blocks (int): The number of blocks.
        """

        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, num_blocks))

        return nn.CellList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""

        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.SequentialCell(
                            nn.Conv2d(in_channels[j],
                                      in_channels[i],
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      has_bias=False),
                            nn.BatchNorm2d(in_channels[i]),
                            UpsampleNearest(scale_factor=2**(j - i))
                        )
                    )
                elif j == i:
                    fuse_layer.append(nn.SequentialCell([IdentityMap()]))
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.SequentialCell(
                                    nn.Conv2d(in_channels[j],
                                              in_channels[j],
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              group=in_channels[j],
                                              has_bias=False,
                                              pad_mode="pad"),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.Conv2d(in_channels[j],
                                              in_channels[i],
                                              kernel_size=1,
                                              stride=1,
                                              padding=0,
                                              has_bias=False),
                                    nn.BatchNorm2d(in_channels[i]),
                                )
                            )
                        else:
                            conv_downsamples.append(
                                nn.SequentialCell(
                                    nn.Conv2d(in_channels[j],
                                              in_channels[j],
                                              kernel_size=3,
                                              stride=2,
                                              padding=1,
                                              group=in_channels[j],
                                              has_bias=False,
                                              pad_mode="pad"),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.Conv2d(in_channels[j],
                                              in_channels[j],
                                              kernel_size=1,
                                              stride=1,
                                              padding=0,
                                              has_bias=False),
                                    nn.BatchNorm2d(in_channels[j]),
                                    nn.ReLU()
                                )
                            )
                    fuse_layer.append(nn.SequentialCell(conv_downsamples))
            fuse_layers.append(nn.CellList(fuse_layer))

        return nn.CellList(fuse_layers)

    def construct(self, x):
        """Construct function."""

        if self.num_branches == 1:
            return [self.layers[0](x[0])]
        output = []
        if self.module_type == 'LITE':
            output = self.layers(x)
        elif self.module_type == 'NAIVE':
            for i in range(self.num_branches):
                x[i] = self.layers[i](x[i])
            output = x

        if self.with_fuse:
            out_fuse = []
            y = []
            for i in range(len(self.fuse_layers)):
                if i == 0:
                    y = output[0]
                else:
                    y = self.fuse_layers[i][0](output[0])
                for j in range(self.num_branches):
                    if i == j:
                        y = y + output[j]
                        if i == 0:
                            output[0] = output[0] + output[0]
                    else:
                        y = y + self.fuse_layers[i][j](output[j])
                        if i == 0:
                            output[0] = output[0] + self.fuse_layers[i][j](output[j])
                out_fuse.append(self.relu(y))
            output = out_fuse
        elif not self.multiscale_output:
            output = [output[0]]
        return output

class LiteHRNet(nn.Cell):
    """
    Lite-HRNet with heatmap head.
    <https://arxiv.org/abs/1904.04514>

    Args:
        extra (dict): detailed configuration for each stage of HRNet, imported from configs.net_configs module.
        in_channels (int): Number of inputs image channels. Default: 3.
        norm_eval (bool): Whether to set norm layers to eval mode.

    Input:
        - **images** (Tensor) - Input image tensors from dataset API.

    Output:
        - **heatmap** (Tensor) - Predicted joint heatmaps.

    Supported Platform:
        ``Ascend`` ``GPU`` ``CPU``

    Example:
        >>> extra = configs.net_configs.litehrnet_18_coco()
        >>> lite_net = LiteHRNet(extra, 3, False)
        >>> pred_heatmap = lite_net(mindspore.Tensor(np.random.rand(16, 3, 256, 192), mindspore.float32))
    """

    def __init__(self,
                 extra,
                 in_channels=3,
                 norm_eval=False
                 ):
        super().__init__()
        self.extra = extra
        self.norm_eval = norm_eval

        self.stem = Stem(
            in_channels,
            stem_channels=self.extra['stem']['stem_channels'],
            out_channels=self.extra['stem']['out_channels'],
            expand_ratio=self.extra['stem']['expand_ratio'])

        self.num_stages = self.extra['num_stages']
        self.stages_spec = self.extra['stages_spec']

        num_channels_last = [
            self.stem.out_channels,
        ]

        self.transitions = nn.CellList()
        self.stages = nn.CellList()

        for i in range(self.num_stages):
            num_channels = self.stages_spec['num_channels'][i]
            num_channels = [num_channels[i] for i in range(len(num_channels))]
            trans = self._make_transition_layer(num_channels_last, num_channels)
            self.transitions.append(trans)
            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, multiscale_output=True)
            self.stages.append(stage)


        self.with_head = self.extra['with_head']
        if self.with_head:
            self.head_layer = IterativeHead(in_channels=num_channels_last)

        self.heat_map = LiteTopDownSimpleHeatMap(extra["final_cfg"])

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """
        Make transition layer.

        Args:
            num_channels_pre_layer (int): Channel size of the previous layer.
            num_channels_cur_layer (int): Channel size of the output.
        """

        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.SequentialCell(
                            nn.Conv2d(num_channels_pre_layer[i],
                                      num_channels_pre_layer[i],
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      group=num_channels_pre_layer[i],
                                      has_bias=False,
                                      pad_mode="pad"),
                            nn.BatchNorm2d(num_channels_pre_layer[i]),
                            nn.Conv2d(num_channels_pre_layer[i],
                                      num_channels_cur_layer[i],
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      has_bias=False),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU()
                        )
                    )
                else:
                    transition_layers.append(nn.SequentialCell([IdentityMap()]))
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.SequentialCell(
                            nn.Conv2d(in_channels,
                                      in_channels,
                                      kernel_size=3,
                                      stride=2,
                                      padding=1,
                                      group=in_channels,
                                      has_bias=False,
                                      pad_mode="pad"),
                            nn.BatchNorm2d(in_channels),
                            nn.Conv2d(in_channels,
                                      out_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0,
                                      has_bias=False),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU()
                        )
                    )
                transition_layers.append(nn.SequentialCell(conv_downsamples))

        return nn.CellList(transition_layers)

    def _make_stage(self,
                    stages_spec,
                    stage_index,
                    in_channels,
                    multiscale_output=True):
        """
        Make stage.

        Args:
            stages_spec (dict): Configures for each stage.
            stage_index (int): Current stage index.
            in_channels (int): Input channel size.
            multiscale_output (bool): Whether outputs have different resolution.
        """

        num_modules = stages_spec['num_modules'][stage_index]
        num_branches = stages_spec['num_branches'][stage_index]
        num_blocks = stages_spec['num_blocks'][stage_index]
        reduce_ratio = stages_spec['reduce_ratios'][stage_index]
        with_fuse = stages_spec['with_fuse'][stage_index]
        module_type = stages_spec['module_type'][stage_index]

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True

            modules.append(
                LiteHRModule(
                    num_branches,
                    num_blocks,
                    in_channels,
                    reduce_ratio,
                    module_type,
                    multiscale_output=reset_multiscale_output,
                    with_fuse=with_fuse))
            in_channels = modules[-1].in_channels

        return nn.SequentialCell(modules), in_channels



    def construct(self, x):
        """Construct function."""

        x = self.stem(x)
        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            for j in range(self.stages_spec['num_branches'][i]):
                if self.transitions[i][j]:
                    if j >= len(y_list):
                        x_list.append(self.transitions[i][j](y_list[-1]))
                    else:
                        x_list.append(self.transitions[i][j](y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = self.stages[i](x_list)

        x = y_list
        if self.with_head:
            x = self.head_layer(x)
        pred_heatmap = self.heat_map(x[0])
        return pred_heatmap

    def train(self, mode=True):
        """
        Convert the model into training mode.

        Args:
            mode (bool): Whether set the model into training mode. Default: True.
        """

        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
