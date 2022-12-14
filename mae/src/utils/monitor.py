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
# ==============================================================================
"""
Monitor for monitoring training status and loss value.
"""

import time

from mindspore.common.tensor import Tensor
from mindspore.train.callback._callback import Callback

import numpy as np


class StateMonitor(Callback):
    """
    Training status monitoring class.

    Args:
        data_size (int): Data size.
        tot_batch_size (int): Total batch size.
        eval_interval (str): Eval interval way.
        eval_offset (str): Eval offset.
        eval_engine (str): Eval engine.
        logger (logger): Logger.
    """

    def __init__(self, data_size, tot_batch_size=None,
                 eval_interval=None, eval_offset=None,
                 eval_engine=None, logger=None):
        super(StateMonitor, self).__init__()
        self.data_size = data_size
        self.tot_batch_size = tot_batch_size
        self.epoch_num = 0
        self.loss = 0
        self.eval_interval = eval_interval
        self.eval_offset = eval_offset
        self.eval_engine = eval_engine
        self.best_acc = -1
        self.best_acc_top5 = -1
        self.best_i2t_recall = -1
        self.best_t2i_recall = -1
        self.mean_fps = 0.0
        self.print = print
        self.epoch_time = 0
        if logger is not None:
            self.print = logger

    def step_end(self, run_context):
        """
        Calculate loss after a step of the model.

        Args:
            run_context (context): The context of the training runtime.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = np.mean(loss.asnumpy())

        self.loss = loss

    def epoch_begin(self, run_context):
        """The epoch recording time at the beginning."""
        print(run_context)
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        """Print the corresponding situation information after each epoch."""
        print(run_context)
        epoch_seconds = (time.time() - self.epoch_time)
        per_step_seconds = epoch_seconds / self.data_size

        print_str = "epoch[{}]".format(self.epoch_num)
        print_str += ', epoch time: {:.2f}s'.format(epoch_seconds)
        print_str += ', per step time: {:.4f}s'.format(per_step_seconds)
        print_str += ', loss={:.6f}'.format(self.loss)

        if self.tot_batch_size:
            # Total number of batches processed per step (batch size * device num)
            fps = self.tot_batch_size * self.data_size / epoch_seconds
            self.mean_fps = (self.mean_fps * self.epoch_num + fps) / (self.epoch_num + 1)
            print_str += ', fps={:.2f}'.format(fps)

        if (self.epoch_num + 1) % self.eval_interval == self.eval_offset:
            # Perform verification, print out information related to accuracy, recall, cost, etc.
            eval_start = time.time()
            self.eval_engine.eval()
            output = self.eval_engine.get_result()
            eval_seconds = time.time() - eval_start
            if output:
                if isinstance(output, list):
                    print_str += ', top1 accuracy={:.6f}'.format(float(output[0]))
                    print_str += ', top5 accuracy={:.6f}'.format(float(output[1]))
                    print_str += ', i2t_recall={:.6f}'.format(float(output[2]))
                    print_str += ', t2i_recall={:.6f}'.format(float(output[3]))
                    print_str += ', eval_cost={:.2f}'.format(eval_seconds)

                    # Update the value of the best parameter
                    if float(output[0]) > self.best_acc:
                        self.best_acc = float(output[0])
                    if float(output[1]) > self.best_acc_top5:
                        self.best_acc_top5 = float(output[1])
                    if float(output[2]) > self.best_i2t_recall:
                        self.best_i2t_recall = float(output[2])
                    if float(output[3]) > self.best_t2i_recall:
                        self.best_t2i_recall = float(output[3])
                else:
                    print_str += ', accuracy={:.6f}'.format(float(output))
                    print_str += ', eval_cost={:.2f}'.format(eval_seconds)

                    if float(output) > self.best_acc:
                        self.best_acc = float(output)

        self.print(print_str)
        self.epoch_num += 1


class LossMonitor(Callback):
    """
    Monitor the loss value during training.
    If the loss value is NAN or INF, training will be terminated.
    If the value of per_print_times is 0, no loss value will be printed.

    Args:
        per_print_times (int): How many steps to print the loss value once.
        In sink mode, it will print the loss at the nearest step. Default value: 1
        log (logger): Logger.
    """

    def __init__(self, per_print_times=1, log=None):
        super(LossMonitor, self).__init__()
        if not isinstance(per_print_times, int) or per_print_times < 0:
            raise ValueError("The argument 'per_print_times' must be int and >= 0, "
                             "but got {}".format(per_print_times))
        self._per_print_times = per_print_times
        self._last_print_time = 0
        self.log = log

    def step_end(self, run_context):
        """
        Print the training loss at the end of the step.

        Args:
            run_context (context): The context of the training runtime.
        """
        cb_params = run_context.original_args()

        loss = cb_params.net_outputs[0].asnumpy()

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))
        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            self.log.info("epoch: %s step: %s, loss is %s" % (cb_params.cur_epoch_num, cur_step_in_epoch, loss))
