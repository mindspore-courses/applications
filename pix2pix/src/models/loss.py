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
# ===========================================================================
""" Define losses."""

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import functional as opsf
import mindspore.ops.operations as opsp
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean, _get_parallel_mode)
from mindspore.context import ParallelMode
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore.nn.loss.loss import LossBase


class SigmoidCrossEntropyWithLogits(LossBase):
    """
    Defining Sigmoid Cross Entropy Loss as Loss Function.

    Inputs:
        -**data** (Tensor) - Tensor of image data.
        -**label** (Tensor) - Tensor of image label.

    Outputs:
        SigmoidCrossEntropy loss function.
    """

    def __init__(self):
        super(SigmoidCrossEntropyWithLogits, self).__init__()
        self.cross_entropy = opsp.SigmoidCrossEntropyWithLogits()

    def construct(self, data, label):
        x = self.cross_entropy(data, label)
        return self.get_loss(x)

class LossD(LossBase):
    """
    Define discriminator loss

    Args:
        config (class): Option class.
        reduction (str): Return loss of the samples. Default: "mean".

    Inputs:
        -**pred1** (Tensor) - predict image1.
        -**pred0** (Tensor) - predict image0.

    Outputs:
        discriminator loss.
    """

    def __init__(self, config, reduction="mean"):    # Return the averaging loss of the samples
        super(LossD, self).__init__(reduction)
        self.sig = SigmoidCrossEntropyWithLogits()
        self.ones = ops.OnesLike()
        self.zeros = ops.ZerosLike()
        self.lambda_dis = config.lambda_dis

    def construct(self, pred1, pred0):
        loss = self.sig(pred1, self.ones(pred1)) + self.sig(pred0, self.zeros(pred0))
        dis_loss = loss * self.lambda_dis
        return dis_loss


class WithLossCellD(nn.Cell):
    """
    Define WithLossCellD to connect the network and Loss.

    Args:
        backbone (Cell): backbone of loss network.
        loss_fn (Cell): init loss function.

    Inputs:
        -**reala** (Tensor) - real label a.
        -**realb** (Tensor) - real label b.

    Outputs:
        connected loss function.
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCellD, self).__init__(auto_prefix=True)
        self.net_discriminator = backbone.net_discriminator
        self.net_generator = backbone.net_generator
        self._loss_fn = loss_fn

    def construct(self, reala, realb):
        fakeb = self.net_generator(reala)
        pred1 = self.net_discriminator(reala, realb)
        pred0 = self.net_discriminator(reala, fakeb)
        return self._loss_fn(pred1, pred0)


class LossG(LossBase):
    """
    Define generator loss.

    Args:
        config (class): Option class.
        reduction (str): Return loss of the samples. Default: "mean".

    Inputs:
        -**fakeb** (Tensor) - generate fake image.
        -**realb** (Tensor) - real image.
        -**pred0** (Tensor) - predict image.

    Outputs:
        generator loss.
    """

    def __init__(self, config, reduction="mean"):   #reduction="mean": Return the averaging loss of the samples
        super(LossG, self).__init__(reduction)
        self.sig = SigmoidCrossEntropyWithLogits()
        self.l1_loss = nn.L1Loss()
        self.ones = ops.OnesLike()
        self.lambda_gan = config.lambda_gan
        self.lambda_l1 = config.lambda_l1

    def construct(self, fakeb, realb, pred0):
        loss_1 = self.sig(pred0, self.ones(pred0))
        loss_2 = self.l1_loss(fakeb, realb)
        loss = loss_1 * self.lambda_gan + loss_2 * self.lambda_l1
        return loss


class WithLossCellG(nn.Cell):
    """
    Define WithLossCellG to connect the network and Loss.

    Args:
        backbone (Cell): backbone of loss network.
        loss_fn (Cell): init loss function.

    Inputs:
        -**reala** (Tensor) - real label a.
        -**realb** (Tensor) - real label b.

    Outputs:
        connected loss function.
    """

    def __init__(self, backbone, loss_fn):
        super(WithLossCellG, self).__init__(auto_prefix=True)
        self.net_discriminator = backbone.net_discriminator
        self.net_generator = backbone.net_generator
        self._loss_fn = loss_fn

    def construct(self, reala, realb):
        fakeb = self.net_generator(reala)
        pred0 = self.net_discriminator(reala, fakeb)
        return self._loss_fn(fakeb, realb, pred0)


class TrainOneStepCell(nn.Cell):
    """
    Define TrainOneStepCell to encapsulate the training of the discriminator and generator together.

    Args:
        loss_netd (Cell): loss network of discriminator.
        loss_netg (Cell): loss network of generator.
        optimizerd (Union[Cell]): optimizer that updates discriminator network parameters.
        optimizerg (Union[Cell]): optimizer that updates generator network parameters.
        sens (numbers.Number): Input to backpropagation, scaling factor. Default: 1.
        auto_prefix (bool): whether auto prefix. Default: True.

    Inputs:
        -**reala** (Tensor) - real label a.
        -**realb** (Tensor) - real label b.

    Outputs:
        d_res, train generator out output.
        g_res, train discriminator output.
    """

    def __init__(self, loss_netd, loss_netg, optimizerd, optimizerg, sens=1, auto_prefix=True):
        super(TrainOneStepCell, self).__init__(auto_prefix=auto_prefix)
        self.loss_net_d = loss_netd
        self.loss_net_d.set_grad()
        self.loss_net_d.add_flags(defer_inline=True)

        self.loss_net_g = loss_netg
        self.loss_net_g.set_grad()
        self.loss_net_g.add_flags(defer_inline=True)

        self.weights_g = optimizerg.parameters
        self.optimizerg = optimizerg
        self.weights_d = optimizerd.parameters
        self.optimizerd = optimizerd

        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

        # Parallel processing
        self.reducer_flag = False
        self.grad_reducer_g = opsf.identity
        self.grad_reducer_d = opsf.identity
        self.parallel_mode = _get_parallel_mode()
        if self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = _get_gradients_mean()
            degree = _get_device_num()
            self.grad_reducer_g = DistributedGradReducer(self.weights_g, mean, degree)
            self.grad_reducer_d = DistributedGradReducer(self.weights_d, mean, degree)

    def set_sens(self, value):
        self.sens = value

    def construct(self, reala, realb):
        """Define TrainOneStepCell."""
        d_loss = self.loss_net_d(reala, realb)
        g_loss = self.loss_net_g(reala, realb)
        d_sens = ops.Fill()(ops.DType()(d_loss), ops.Shape()(d_loss), self.sens)
        d_grads = self.grad(self.loss_net_d, self.weights_d)(reala, realb, d_sens)
        d_res = ops.depend(d_loss, self.optimizerd(d_grads))
        g_sens = ops.Fill()(ops.DType()(g_loss), ops.Shape()(g_loss), self.sens)
        g_grads = self.grad(self.loss_net_g, self.weights_g)(reala, realb, g_sens)
        g_res = ops.depend(g_loss, self.optimizerg(g_grads))
        return d_res, g_res
