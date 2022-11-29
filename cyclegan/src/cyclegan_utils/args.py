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
"""get args."""

from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size


def get_args(args):
    """
    Define the common options that are used in both training and test.

    Args:
        args (class): option class.

    Returns:
        Args.
    """

    if args.device_num > 1:

        context.set_context(mode=context.GRAPH_MODE, device_target=args.platform, save_graphs=args.save_graphs)
        context.reset_auto_parallel_context()
        context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                          device_num=args.device_num)
        init()
        args.rank = get_rank()
        args.group_size = get_group_size()
    else:
        context.set_context(mode=context.GRAPH_MODE, device_target=args.platform,
                            save_graphs=args.save_graphs, device_id=args.device_id)
        args.rank = 0
        args.device_num = 1

    if args.platform == "GPU":
        context.set_context(enable_graph_kernel=True)

    if args.platform == "Ascend":
        args.pad_mode = "CONSTANT"

    if args.phase != "train" and (args.g_a_ckpt is None or args.g_b_ckpt is None):
        raise ValueError('Must set g_a_ckpt and g_b_ckpt in predict phase!')

    if args.batch_size == 1:
        args.norm_mode = "instance"

    if args.max_dataset_size is None:
        args.max_dataset_size = float("inf")

    args.n_epochs = min(args.max_epoch, args.n_epochs)
    args.n_epochs_decay = args.max_epoch - args.n_epochs
    return args
