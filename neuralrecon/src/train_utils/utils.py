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
"""Utils for MindSpore training"""

import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P


class InternalCallbackParam(dict):
    """Internal callback object's parameters."""

    def __getattr__(self, para_name):
        return self[para_name]

    def __setattr__(self, para_name, para_value):
        self[para_name] = para_value


class NeuralReconWithLossCell(nn.Cell):
    """
    NeuralRecon with loss cell.
    """

    def __init__(self, network):
        super(NeuralReconWithLossCell, self).__init__()
        self.network = network

    def construct(self, item):
        _, loss_dict = self.network(item)
        loss = loss_dict['total_loss']
        return loss


class TrainingWrapper(nn.TrainOneStepWithLossScaleCell):
    """
    Training wrapper
    """

    def __init__(self, network, optimizer, sens=1.0):
        scaling_sens = sens
        if isinstance(scaling_sens, (int, float)):
            scaling_sens = ms.Tensor(scaling_sens, ms.float32)
        super(TrainingWrapper, self).__init__(network, optimizer, scaling_sens)
        self.sens = sens

    def construct(self, item):
        """
        Construct method.
        """

        weights = self.weights
        scaling_sens = self.scale_sense
        loss = self.network(item)
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)

        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(item)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        cond = self.get_overflow_status(status, grads)
        ret = (loss, cond, sens)
        self.optimizer(grads)
        return ret


def get_param_groups(network):
    """
    Get param groups
    """

    decay_params = []
    no_decay_params = []
    for x in network.trainable_params():
        parameter_name = x.name
        if parameter_name.endswith('.bias'):
            # all bias not using weight decay
            no_decay_params.append(x)
        elif parameter_name.endswith('.gamma'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        elif parameter_name.endswith('.beta'):
            # bn weight bias not using weight decay, be carefully for now x not include BN
            no_decay_params.append(x)
        else:
            decay_params.append(x)

    return [{'params': no_decay_params, 'weight_decay': 0.0}, {'params': decay_params}]
