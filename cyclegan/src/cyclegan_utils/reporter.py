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
"""Reporter class."""

import os
import time
import logging
from datetime import datetime

from mindspore.train.serialization import save_checkpoint

from .tools import save_image


class Reporter(logging.Logger):
    """
    This class includes several functions that can save images/checkpoints and print/save logging information.

    Args:
        args (class): Option class.
    """

    def __init__(self, args):
        super(Reporter, self).__init__("cyclegan")
        self.log_dir = args.outputs_log
        self.imgs_dir = args.outputs_imgs
        self.ckpts_dir = args.outputs_ckpt
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        if not os.path.exists(self.imgs_dir):
            os.makedirs(self.imgs_dir, exist_ok=True)
        if not os.path.exists(self.ckpts_dir):
            os.makedirs(self.ckpts_dir, exist_ok=True)
        self.rank = args.rank
        self.save_checkpoint_epochs = args.save_checkpoint_epochs
        self.save_imgs = args.save_imgs

        # console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        self.addHandler(console)

        # file handler
        log_name = datetime.now().strftime('%Y-%m-%d_time_%H_%M_%S') + '_rank_{}.log'.format(self.rank)
        self.log_fn = os.path.join(self.log_dir, log_name)
        fh = logging.FileHandler(self.log_fn)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.addHandler(fh)
        self.step = 0
        self.epoch = 0
        self.dataset_size = args.dataset_size // args.device_num
        self.device_num = args.device_num
        self.print_iter = args.print_iter
        self.g_loss = []
        self.d_loss = []

    def info(self, msg, *args, **kwargs):
        """
        get the log.

        Args:
            msg(str): type of message.
            *args(class): Option class.
            **kwargs(class): Option class.
        """

        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def epoch_start(self):
        """
        This can get training start and save time.
        """
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()
        self.step = 0
        self.epoch += 1
        self.g_loss = []
        self.d_loss = []

    def step_end(self, res_g, res_d):
        """
        This can print log when step end.

        Args:
            res_g(numpy array / Tensor): res of generator.
            res_d(numpy array / Tensor): res of discriminator.
        """

        self.step += 1
        loss_d = float(res_d.asnumpy())
        res = []
        for item in res_g[2:]:
            res.append(float(item.asnumpy()))
        self.g_loss.append(res[0])
        self.d_loss.append(loss_d)
        if self.step % self.print_iter == 0:
            step_cost = (time.time() - self.step_start_time) * 1000 / self.print_iter
            losses = "G_loss: {:.2f}, D_loss:{:.2f}, loss_G_A: {:.2f}, loss_G_B: {:.2f}, loss_C_A: {:.2f},"\
                     "loss_C_B: {:.2f}, loss_idt_A: {:.2f}, loss_idt_B：{:.2f}".format(
                         res[0], loss_d, res[1], res[2], res[3], res[4], res[5], res[6])
            self.info("Epoch[{}] [{}/{}] step cost: {:.2f} ms, {}".format(
                self.epoch, self.step, self.dataset_size, step_cost, losses))
            self.step_start_time = time.time()

    def epoch_end(self, net):
        """
        This will print log and save checkpoints when epoch end.

        Args:
            net(Cell): training network.
        """

        epoch_cost = (time.time() - self.epoch_start_time) * 1000
        per_step_time = epoch_cost / self.dataset_size
        mean_loss_g = sum(self.g_loss) / self.dataset_size
        mean_loss_d = sum(self.d_loss) / self.dataset_size
        self.info("Epoch [{}] total cost: {:.2f} ms, per step: {:.2f} ms, G_loss: {:.2f}, D_loss: {:.2f}".format(
            self.epoch, epoch_cost, per_step_time, mean_loss_d, mean_loss_g))

        if self.epoch % self.save_checkpoint_epochs == 0:
            save_checkpoint(net.g.generator.g_a, os.path.join(self.ckpts_dir, f"g_a_{self.epoch}.ckpt"))
            save_checkpoint(net.g.generator.g_b, os.path.join(self.ckpts_dir, f"g_b_{self.epoch}.ckpt"))
            save_checkpoint(net.g.d_a, os.path.join(self.ckpts_dir, f"d_a_{self.epoch}.ckpt"))
            save_checkpoint(net.g.d_b, os.path.join(self.ckpts_dir, f"d_b_{self.epoch}.ckpt"))

    def visualizer(self, img_a, img_b, fake_a, fake_b):
        """
        This will save the generated picture.

        Args:
            img_a(image): The image of domain A.
            img_b(image): The image of domain B.
            fake_a(image): The image of generator generate from domain B.
            fake_b(image): The image of generator generate from domain A.
        """

        if self.save_imgs and self.step % self.dataset_size == 0:
            save_image(img_a, os.path.join(self.imgs_dir, f"{self.epoch}_img_a.jpg"))
            save_image(img_b, os.path.join(self.imgs_dir, f"{self.epoch}_img_b.jpg"))
            save_image(fake_a, os.path.join(self.imgs_dir, f"{self.epoch}_fake_a.jpg"))
            save_image(fake_b, os.path.join(self.imgs_dir, f"{self.epoch}_fake_b.jpg"))

    def start_predict(self, direction):
        """
        This can start predict and get time.

        Args:
            direction(str): start predict A to B or B to A.
        """

        self.predict_start_time = time.time()
        self.direction = direction
        self.info('==========start predict %s===============', self.direction)

    def end_predict(self):
        """
        get cost time of predict.
        """
        cost = (time.time() - self.predict_start_time) * 1000
        per_step_cost = cost / self.dataset_size
        self.info('total {} imgs cost {:.2f} ms, per img cost {:.2f}'.format(self.dataset_size, cost, per_step_cost))
        self.info('==========end predict %s===============\n', self.direction)
