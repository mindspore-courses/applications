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
"""train step for nerf"""

import mindspore as md
from mindspore import nn

from engine import metrics

__all__ = ["train_net", "RendererWithCriterion"]


def train_net(config, iter_, train_renderer, optimizer, rays, gt):
    """
    Train a network.

    Args:
        config (Config): Configuration.
        iter_ (int): Current iterations.
        renderer (Callable): A volume renderer.
        optimizer (Optimizer): A network optimizer.
        rays (Tensor): A batch of rays for training. (#rays * #samples, 6)
        gt (Tensor): The ground truth.

    Returns:
        Tuple of 2 float, recorded metrics.

        - **loss** (float) - Loss to be recorded.
        - **psnr** (float) - PSNR to be recorded.
    """
    loss = train_renderer(rays, gt)

    # Update learning rate
    decay_rate = 0.1
    decay_steps = config.l_rate_decay * 1000
    new_l_rate = config.l_rate * (decay_rate**(iter_ / decay_steps))
    optimizer.learning_rate = md.Parameter(new_l_rate)

    return float(loss), float(metrics.psnr_from_mse(loss))


class RendererWithCriterion(nn.Cell):
    """
    Renderer with criterion.

    Args:
        renderer (nn.Cell): Renderer.
        loss_fn (nn.Cell, optional): Loss function. Default: nn.MSELoss().

    Inputs:
        - **rays** (Tensor) - Rays tensor.
        - **gt** (Tensor) - Ground truth tensor.

    Outputs:
        Tensor, loss for one forward pass.
    """
    def __init__(self, renderer, loss_fn=nn.MSELoss()):
        """Renderer with criterion."""
        super().__init__()
        self.renderer = renderer
        self.loss_fn = loss_fn

    def construct(self, rays, gt):
        """Renderer Trainer construct."""
        rgb_map_fine, rgb_map_coarse = self.renderer(rays)
        return self.loss_fn(rgb_map_fine, gt) + self.loss_fn(rgb_map_coarse, gt)

    def backbone_network(self):
        """Return renderer."""
        return self.renderer
