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
"""tools for loss."""

import mindspore as ms
import mindspore.ops as ops
from mindspore.ops import constexpr
from mindspore import nn, Tensor
from mindspore.ops.composite import GradOperation


def gan_wgan_loss(pos, neg):
    """
    Return the adversarial loss of generator and discriminator.

    Args:
        pos(Tensor): discriminator output of real images.
        neg(Tensor): discriminator output of generated images.

    Return:
        g_loss: Adversarial loss of generator.
        d_loss: Adversarial loss of discriminator.
    """

    d_loss = ops.ReduceMean(False)(neg) - ops.ReduceMean(False)(pos)
    g_loss = -ops.ReduceMean(False)(neg)
    return g_loss, d_loss


@constexpr
def generate_tensor0():
    """
    Generate tensor with the value of 0.

    Return:
        t0: Tensor.
    """

    t0 = Tensor(0, ms.float32)
    return t0


@constexpr
def generate_tensor1():
    """
    Generate tensor with the value of 1.

    Return:
        t1: Tensor.
    """

    t1 = Tensor(1, ms.float32)
    return t1


def random_interpolates(pos, neg):
    """
    Generate interpolations between real images and generated images.

    Args:
        pos(Tensor): real images.
        neg(Tensor): generated images.

    Return:
        x_hat: Interpolated image.
    """

    minval = generate_tensor0()
    maxval = generate_tensor1()
    epsilon = ops.uniform((pos.shape[0], 1, 1, 1), minval, maxval, dtype=ms.float32)
    x_hat = pos + epsilon * (neg - pos)
    return x_hat


class GradientsPenalty(nn.Cell):
    """
    Return the WGAN-gp loss of discriminator.

    Args:
        net_d(cell): Discriminator.

    Return:
        gradients_penalty: WGAN-gp loss.
    """

    def __init__(self, net_d):
        super(GradientsPenalty, self).__init__()
        self.sqrt = ops.Sqrt()
        self.reducesum = ops.ReduceSum()
        self.square = ops.Square()
        self.reducemean = ops.ReduceMean()
        self.gradients = GradOperation(get_all=False)(net_d)

    def construct(self, interpolates_global):
        grad_d_x_hat = self.gradients(interpolates_global)
        slopes = self.sqrt(self.reducesum(self.square(grad_d_x_hat), [1, 2, 3]))
        gradients_penalty = self.reducemean((slopes - 1) ** 2)
        return gradients_penalty
