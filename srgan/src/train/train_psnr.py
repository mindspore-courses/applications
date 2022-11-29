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
"""Train Srresnet network"""

import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import context
from mindspore.context import ParallelMode
import mindspore.ops.functional as F
from mindspore.parallel._auto_parallel_context import auto_parallel_context
from mindspore.communication.management import get_group_size

__all__ = ["TrainOnestepPSNR"]

class TrainOnestepPSNR(nn.Cell):
    """
    Encapsulation class of Cycle GAN generator network training.
    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        psnr (nn.Cell): Generator with loss function.
        optimizer (nn.Adam): Optimizer used in training step.
        sens (float): The sensitive for backpropagation. Default: 1.0.

    Inputs:
        - **hr_img** (Tensor) - The high-resolution image.
          The input shape must be (batchsize, num_channels, height, width).
        - **lr_img** (Tensor) - The low-resolution image.
          The input shape must be (batchsize, num_channels, height, width).

    Outputs:
        - **F.depend(psnr_loss, self.optimizer(grads))** (Tensor) - The MSE loss.
          The output has the shape (batchsize, loss_value).

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> import numpy as np
        >>> import mindspore.common.dtype as mstype
        >>> import mindspore.nn as nn
        >>> import mindspore.ops as ops
        >>> from mindspore import context
        >>> from mindspore.context import ParallelMode
        >>> import mindspore.ops.functional as F
        >>> from mindspore import Tensor
        >>> from mindspore.parallel._auto_parallel_context import auto_parallel_context
        >>> from mindspore.communication.management import get_group_size
        >>> from model.generator import get_generator
        >>> from loss.psnr_loss import PSNRLoss
        >>> generator = get_generator(4, 0.02)
        >>> psnr_loss = PSNRLoss(generator)
        >>> psnr_optimizer = nn.Adam(generator.trainable_params(), 1e-4)
        >>> train_psnr = TrainOnestepPSNR(psnr_loss, psnr_optimizer)
        >>> hr_img = Tensor(np.zeros([16, 3, 96, 96]),mstype.float32)
        >>> lr_img = Tensor(np.zeros([16, 3, 24, 24]),mstype.float32)
        >>> train_psnr.set_train()
        >>> result = train_psnr(hr_img, lr_img)
        >>> print(result)
        Tensor(shape=[], dtype=Float32, value= 0)
    """
    def __init__(self, psnr, optimizer, sens=1.0):
        super(TrainOnestepPSNR, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.psnr = psnr
        self.psnr.set_grad()
        self.psnr.set_train()
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
        psnr_loss = self.psnr(hr_img, lr_img)
        sens = ops.Fill()(ops.DType()(psnr_loss), ops.Shape()(psnr_loss), self.sens)
        grads = self.grad(self.psnr, weights)(hr_img, lr_img, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return F.depend(psnr_loss, self.optimizer(grads))
