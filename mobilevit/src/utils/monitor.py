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
""" Init for base architecture engine monitor register. """

import time

from typing import Optional, Union, Iterable
import numpy as np

import mindspore as ms
from mindspore.train.callback import Callback

__all__ = ["LossMonitor"]


class LossMonitor(Callback):
    """
    Loss Monitor for classification.

    Args:
        lr_init (Union[float, Iterable], optional): The learning rate schedule. Default: None.
        per_print_times (int): Every how many steps to print the log information. Default: 1.

    Examples:
        >>> from mindvision.engine.callback import LossMonitor
        >>> lr = [0.01, 0.008, 0.006, 0.005, 0.002]
        >>> monitor = LossMonitor(lr_init=lr, per_print_times=100)
    """

    def __init__(self,
                 lr_init: Optional[Union[float, Iterable]] = None,
                 per_print_times: int = 1):
        super(LossMonitor, self).__init__()
        self.lr_init = lr_init
        self.per_print_times = per_print_times
        self.last_print_time = 0

    # pylint: disable=unused-argument
    def epoch_begin(self, run_context):
        """
        Record time at the beginning of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """
        Print training info at the end of epoch.

        Args:
            run_context (RunContext): Context of the process running.
        """
        callback_params = run_context.original_args()
        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / callback_params.batch_num
        print(f"Epoch time: {epoch_mseconds:5.3f} ms, "
              f"per step time: {per_step_mseconds:5.3f} ms, "
              f"avg loss: {np.mean(self.losses):5.3f}", flush=True)

    # pylint: disable=unused-argument
    def step_begin(self, run_context):
        """
        Record time at the beginning of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        self.step_time = time.time()

    # pylint: disable=missing-docstring
    def step_end(self, run_context):
        """
        Print training info at the end of step.

        Args:
            run_context (RunContext): Context of the process running.
        """
        callback_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        loss = callback_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], ms.Tensor) and isinstance(loss[0].asnumpy(), np.ndarry):
                loss = loss[0]

        if isinstance(loss, ms.Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        self.losses.append(loss)
        cur_step_in_epoch = (callback_params.cur_step_num - 1) % callback_params.batch_num + 1

        # Boundary check.
        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(f"Invalid loss, terminate training.")

        def print_info():
            lr_output = self.lr_init[callback_params.cur_step_num - 1] if isinstance(self.lr_init,
                                                                                     list) else self.lr_init
            print(f"Epoch:[{(callback_params.cur_epoch_num - 1):3d}/{callback_params.epoch_num:3d}], "
                  f"step:[{cur_step_in_epoch:5d}/{callback_params.batch_num:5d}], "
                  f"loss:[{loss:5.3f}/{np.mean(self.losses):5.3f}], "
                  f"time:{step_mseconds:5.3f} ms, "
                  f"lr:{lr_output:5.5f}", flush=True)

        if (callback_params.cur_step_num - self.last_print_time) >= self.per_print_times:
            self.last_print_time = callback_params.cur_step_num
            print_info()
