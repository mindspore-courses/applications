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
"""Cycle GAN network."""

import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size

from .utils.init_weight import init_weights
from .discriminator import Discriminator
from .generator import ResNetGenerator
from .withloss import WithLossCell


def get_generator(args):
    """
    This will implement the CycleGAN model, for learning image-to-image translation without paired data.

    Args:
        in_planes (int): in_planes. Default: 3.
        ngf (int): generator model filter numbers. Default: 64.
        gl_num (int): generator model residual block numbers. Default: 9.
        alpha (float): leakyrelu slope. Default: 0.02.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance". Default: "batch".
        dropout (bool): Use dropout or not. Default: False.
        pad_mode(str): the type of Pad. The optional values are 'CONSTANT', 'REFLECT', 'SYMMETRIC'. Default: "CONSTANT".
        init_type(str): network initialization. The optional values are 'normal', 'xavier'. Default: 'normal'.
        init_gain(float): scaling factor for normal, xavier and orthogonal. Default: 0.02.

    Returns:
        nn.Cell.
    """

    net = ResNetGenerator(in_planes=args.in_planes, ngf=args.ngf, n_layers=args.gl_num,
                          alpha=args.slope, norm_mode=args.norm_mode, dropout=args.need_dropout,
                          pad_mode=args.pad_mode)
    init_weights(net, args.init_type, args.init_gain)

    return net


def get_discriminator(args):
    """
    This will return discriminator by args.

    Args:
        in_planes (int): in_planes. Default: 3.
        ndf (int): discriminator model filter numbers. Default: 64.
        gl_num (int): generator model residual block numbers. Default: 9.
        alpha (float): leakyrelu slope. Default: 0.02.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance". Default: "batch".
        init_type(str): network initialization. The optional values are 'normal', 'xavier'. Default: 'normal'.
        init_gain(float): scaling factor for normal, xavier and orthogonal. Default: 0.02.

    Returns:
        nn.Cell.
    """

    net = Discriminator(in_planes=args.in_planes, ndf=args.ndf, n_layers=args.dl_num,
                        alpha=args.slope, norm_mode=args.norm_mode)
    init_weights(net, args.init_type, args.init_gain)
    return net


class Generator(nn.Cell):
    """
    Generator of CycleGAN, return fake_a, fake_b, rec_a, rec_b, identity_a and identity_b.

    Args:
        g_a (Cell): The generator network of domain a to domain b.
        g_b (Cell): The generator network of domain b to domain a.
        use_identity (bool): Use identity loss or not. Default: True.

    Returns:
        Tensors, fake_a, fake_b, rec_a, rec_b, identity_a and identity_b.

    Examples:
        >>> Generator(g_a, g_b)
    """

    def __init__(self, g_a, g_b, use_identity=True):
        super(Generator, self).__init__()
        self.g_a = g_a
        self.g_b = g_b
        self.ones = ops.OnesLike()
        self.use_identity = use_identity

    def construct(self, img_a, img_b):
        """If use_identity, identity loss will be used."""
        fake_a = self.g_b(img_b)
        fake_b = self.g_a(img_a)
        rec_a = self.g_b(fake_b)
        rec_b = self.g_a(fake_a)
        if self.use_identity:
            identity_a = self.g_b(img_a)
            identity_b = self.g_a(img_b)
        else:
            identity_a = self.ones(img_a)
            identity_b = self.ones(img_b)
        return fake_a, fake_b, rec_a, rec_b, identity_a, identity_b


class TrainOneStepG(nn.Cell):
    """
    Encapsulation class of Cycle GAN generator network training.

    Args:
        g (Cell): Generator with loss Cell. Note that loss function should have been added.
        generator (Cell): Generator of CycleGAN.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """

    def __init__(self, g, generator, optimizer, sens=1.0):
        super(TrainOneStepG, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.g = g
        self.g.set_grad()
        self.g.set_train()
        self.g.d_a.set_grad(False)
        self.g.d_a.set_train(False)
        self.g.d_b.set_grad(False)
        self.g.d_b.set_train(False)
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ms.ParameterTuple(generator.trainable_params())
        self.net = WithLossCell(g)
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, img_a, img_b):
        weights = self.weights
        fake_a, fake_b, lg, lga, lgb, lca, lcb, lia, lib = self.g(img_a, img_b)
        sens = ops.Fill()(ops.DType()(lg), ops.Shape()(lg), self.sens)
        grads_g = self.grad(self.net, weights)(img_a, img_b, sens)
        if self.reducer_flag:
            grads_g = self.grad_reducer(grads_g)

        return fake_a, fake_b, ops.depend(lg, self.optimizer(grads_g)), lga, lgb, lca, lcb, lia, lib


class TrainOneStepD(nn.Cell):
    """
    Encapsulation class of Cycle GAN discriminator network training.

    Args:
        g (Cell): Generator with loss Cell. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """

    def __init__(self, d, optimizer, sens=1.0):
        super(TrainOneStepD, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.d = d
        self.d.set_grad()
        self.d.set_train()
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ms.ParameterTuple(d.trainable_params())
        self.reducer_flag = False
        self.grad_reducer = None
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            if auto_parallel_context().get_device_num_is_set():
                degree = context.get_auto_parallel_context("device_num")
            else:
                degree = get_group_size()
            self.grad_reducer = nn.DistributedGradReducer(optimizer.parameters, mean, degree)

    def construct(self, img_a, img_b, fake_a, fake_b):
        weights = self.weights
        ld = self.d(img_a, img_b, fake_a, fake_b)
        sens_d = ops.Fill()(ops.DType()(ld), ops.Shape()(ld), self.sens)
        grads_d = self.grad(self.d, weights)(img_a, img_b, fake_a, fake_b, sens_d)
        if self.reducer_flag:
            grads_d = self.grad_reducer(grads_d)
        return ops.depend(ld, self.optimizer(grads_d))
