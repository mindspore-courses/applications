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
"""Train SRGAN network"""

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size

__all__ = ["TrainOneStepD"]

class TrainOneStepD(nn.Cell):
    """
    Encapsulation class of Cycle GAN generator network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        d (nn.Cell): Discriminator with loss function.
        optimizer (nn.Adam): Optimizer used in training step.
        sens (float): The sensitive for backpropagation. Default: 1.0.

    Inputs:
        - **hr_img** (Tensor) - The high-resolution image.
          The input shape must be (batchsize, num_channels, height, width).
        - **lr_img** (Tensor) - The low-resolution image.
          The input shape must be (batchsize, num_channels, height, width).

    Outputs:
        - **ld** (Tensor) - The loss of discriminator.
          The output has the shape (batchsize, loss_value).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import context
        >>> from mindspore.context import ParallelMode
        >>> from mindspore.parallel._auto_parallel_context import auto_parallel_context
        >>> from mindspore.communication.management import get_group_size
        >>> from model.discriminator import get_discriminator
        >>> from model.generator import get_generator
        >>> from loss.gan_loss import DiscriminatorLoss
        >>> generator = get_generator(4, 0.02)
        >>> discriminator = get_discriminator(96, 0.02)
        >>> discriminator_loss = DiscriminatorLoss(discriminator, generator)
        >>> discriminator_optimizer = nn.Adam(discriminator.trainable_params(), 1e-4)
        >>> train_discriminator = TrainOneStepD(discriminator_loss, discriminator_optimizer)
        >>> hr_img = Tensor(np.zeros([16, 3, 96, 96]),mstype.float32)
        >>> lr_img = Tensor(np.zeros([16, 3, 24, 24]),mstype.float32)
        >>> train_discriminator.set_train()
        >>> result = train_discriminator(hr_img, lr_img)
        >>> print(result)
        Tensor(shape=[16, 1], dtype=Float32, value=
        [[ 1.38629436e+00],
        [ 1.38629436e+00],
        [ 1.38629436e+00],
        ...
        [ 1.38629436e+00],
        [ 1.38629436e+00],
        [ 1.38629436e+00]])
    """
    def __init__(self, net, optimizer, sens=1.0):
        super(TrainOneStepD, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.d = net
        self.d.set_grad()
        self.d.set_train()
        self.d.generator.set_grad(False)
        self.d.generator.set_train(False)
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
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

    def construct(self, hr_img, lr_img):
        weights = self.weights
        ld = self.d(hr_img, lr_img)
        sens_d = ops.Fill()(ops.DType()(ld), ops.Shape()(ld), self.sens)
        grads_d = self.grad(self.d, weights)(hr_img, lr_img, sens_d)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_d = self.grad_reducer(grads_d)
        self.optimizer(grads_d)
        return ld

class TrainOnestepG(nn.Cell):
    """
    Encapsulation class of Cycle GAN generator network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        g (nn.Cell): Generator with loss function.
        optimizer (nn.Adam): Optimizer used in training step.
        sens (float): Default: 1.0.

    Inputs:
        - **hr_img** (Tensor) - The high-resolution image.
          The input shape must be (batchsize, num_channels, height, width).
        - **lr_img** (Tensor) - The low-resolution image.
          The input shape must be (batchsize, num_channels, height, width).

    Outputs:
        - **lg** (Tensor) - The loss of generator.
          The output has the shape (batchsize, loss_value).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> from mindspore import Tensor
        >>> import mindspore.common.dtype as mstype
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import context
        >>> from mindspore.context import ParallelMode
        >>> from mindspore.parallel._auto_parallel_context import auto_parallel_context
        >>> from mindspore.communication.management import get_group_size
        >>> from model.generator import get_generator
        >>> from model.discriminator import get_discriminator
        >>> from loss.gan_loss import GeneratorLoss
        >>> from vgg19.define import vgg19
        >>> generator = get_generator(4, 0.02)
        >>> discriminator = get_discriminator(96, 0.02)
        >>> vgg = vgg19('./src/vgg19/vgg19.ckpt')
        >>> generator_loss = GeneratorLoss(discriminator, generator, vgg)
        >>> generator_optimizer = nn.Adam(generator.trainable_params(), 1e-4)
        >>> train_generator = TrainOnestepG(generator_loss, generator_optimizer)
        >>> hr_img = Tensor(np.zeros([16, 3, 96, 96]),mstype.float32)
        >>> lr_img = Tensor(np.zeros([16, 3, 24, 24]),mstype.float32)
        >>> train_generator.set_train()
        >>> result = train_generator(hr_img, lr_img)
        >>> print(result)
        [[0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]
        [0.00069315]]
    """
    def __init__(self, g, optimizer, sens=1.0):
        super(TrainOnestepG, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.g = g
        self.g.set_grad()
        self.g.set_train()
        self.g.vgg.set_grad(False)
        self.g.vgg.set_train(False)
        self.g.discriminator.set_grad(False)
        self.g.discriminator.set_train(False)
        self.g.meanshif.set_grad(False)
        self.g.meanshif.set_train(False)
        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = optimizer.parameters
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

    def construct(self, hr_img, lr_img):
        weights = self.weights
        lg = self.g(hr_img, lr_img)
        sens_g = ops.Fill()(ops.DType()(lg), ops.Shape()(lg), self.sens)
        grads_g = self.grad(self.g, weights)(hr_img, lr_img, sens_g)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads_g = self.grad_reducer(grads_g)
        self.optimizer(grads_g)
        return lg
