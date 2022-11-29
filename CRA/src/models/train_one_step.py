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
"""Define train cell."""

from mindspore import nn, ops
import mindspore.ops.functional as F
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean, _get_parallel_mode)
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer


class TrainOneStepD(nn.Cell):
    """
    Encapsulation class of discriminator network training.

    Args:
        d(Cell): discriminator with loss Cell.
        optimizer(Optimizer): Optimizer for updating the weights.
        sens(Number): The adjust parameter.Default: 1.0.

    Return:
        loss_d: Discriminator loss.
    """

    def __init__(self, d, optimizer, sens=1.0):
        super(TrainOneStepD, self).__init__(auto_prefix=True)
        self.optimizer = optimizer
        self.d = d
        self.d.net_d.set_grad()
        self.d.net_d.set_train()
        self.d.net_g.set_grad(False)
        self.d.net_g.set_train(False)
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.reducer_flag = False
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)

    def construct(self, real, x, mask):
        weights = self.weights
        loss_d = self.d(real, x, mask)
        sens_d = self.fill(self.dtype(loss_d), self.shape(loss_d), self.sens)
        grads_d = self.grad(self.d, weights)(real, x, mask, sens_d)
        if self.reducer_flag:
            grads_d = self.grad_reducer(grads_d)
        self.optimizer(grads_d)
        return loss_d


class TrainOneStepG(nn.Cell):
    """
    Encapsulation class of generator network training.

    Args:
        g(Cell): generator with loss Cell.
        optimizer(Optimizer): Optimizer for updating the weights.
        sens(Number): The adjust parameter.Default: 1.0.

    Return:
        loss_g: Generator loss.
    """

    def __init__(self, g, optimizer, sens=1.0):
        super(TrainOneStepG, self).__init__(auto_prefix=True)
        self.optimizer = optimizer
        self.g = g
        self.g.net_g.set_grad()
        self.g.net_g.set_train()
        self.g.net_d.set_grad(False)
        self.g.net_d.set_train(False)
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
        self.reducer_flag = False
        self.fill = ops.Fill()
        self.dtype = ops.DType()
        self.shape = ops.Shape()
        self.grad_reducer = F.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer = DistributedGradReducer(self.weights, mean, degree)

    def construct(self, real, x, mask):
        weights = self.weights
        loss_g = self.g(real, x, mask)
        sens_g = self.fill(self.dtype(loss_g), self.shape(loss_g), self.sens)
        grads_g = self.grad(self.g, weights)(real, x, mask, sens_g)
        if self.reducer_flag:
            grads_g = self.grad_reducer(grads_g)
        self.optimizer(grads_g)
        return loss_g
