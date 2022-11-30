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
"""Loss function for Upsample and encapsulation class of network training."""

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.ops.operations as P
from mindspore import Tensor
from mindspore.ops import stop_gradient
from mindspore.context import ParallelMode
from mindspore.parallel._utils import (_get_device_num, _get_gradients_mean, _get_parallel_mode)


class GeneratorWithLoss(nn.Cell):
    """
    Wrap the network with loss function to return Generator with loss.

    Args:
        generator (nn.Cell): The target generator network to wrap.
        discriminator (nn.Cell): The target discriminator network to wrap.
        vgg_path (str): The path of vgg model checkpoint.
        inpaint_adv_loss_weight (float): The coefficient of adversarial loss.
        l1_loss_weight (float): The coefficient of L1 loss.
        content_loss_weight (float): The coefficient of content loss.
        style_loss_weight (float): The coefficient of style loss.
    """

    def __init__(self, generator, discriminator, vgg_path: str, inpaint_adv_loss_weight: float, l1_loss_weight: float,
                 content_loss_weight: float,
                 style_loss_weight: float):
        super(GeneratorWithLoss, self).__init__(auto_prefix=False)
        self.generator = generator
        self.discriminator = discriminator
        self.inpaint_adv_loss_weight = inpaint_adv_loss_weight
        self.l1_loss_weight = l1_loss_weight
        self.content_loss_weight = content_loss_weight
        self.style_loss_weight = style_loss_weight
        self.adversarial_loss = AdversarialLoss()
        self.l1_loss = nn.L1Loss()
        self.perceptual_loss = PerceptualLoss(vgg_path=vgg_path)
        self.style_loss = StyleLoss(vgg_path)

    def construct(self, images, edges, masks):
        """Forward and return generator loss."""
        loss = 0
        outputs = self.generator(images, edges, masks)
        gen_input_fake = outputs
        gen_fake, _ = self.discriminator(gen_input_fake)
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False) * self.inpaint_adv_loss_weight
        loss += gen_gan_loss
        gen_l1_loss = self.l1_loss(outputs, images) * self.l1_loss_weight / P.ReduceMean()(masks)
        loss += gen_l1_loss
        gen_content_loss = self.perceptual_loss(outputs, images)
        gen_content_loss = gen_content_loss * self.content_loss_weight
        loss += gen_content_loss
        gen_style_loss = self.style_loss(outputs * masks, images * masks)
        gen_style_loss = gen_style_loss * self.style_loss_weight
        loss += gen_style_loss
        return loss


class DiscriminatorWithLoss(nn.Cell):
    """
    Wrap the network with loss function to return Discriminator with loss.

    Args:
        generator (nn.Cell): The target generator network to wrap.
        discriminator (nn.Cell): The target discriminator network to wrap.
    """

    def __init__(self, generator, discriminator):
        super(DiscriminatorWithLoss, self).__init__(auto_prefix=False)
        self.generator = generator
        self.discriminator = discriminator
        self.adversarial_loss = AdversarialLoss()

    def construct(self, images, edges, masks):
        """Forward and return discriminator loss."""
        loss = 0
        outputs = self.generator(images, edges, masks)
        dis_input_real = images
        dis_input_fake = stop_gradient(outputs)
        dis_real, _ = self.discriminator(dis_input_real)
        dis_fake, _ = self.discriminator(dis_input_fake)
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        # dis_real_loss = Tensor(dis_real_loss)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        # dis_fake_loss = Tensor(dis_fake_loss)
        loss += (dis_real_loss + dis_fake_loss) / Tensor(2.0)
        return loss


class TrainOneStepCell(nn.Cell):
    """
    Encapsulation class of GAN generator network training.

    Args:
        generator (nn.Cell): Generator with loss Cell.
        discriminator (nn.Cell): Discriminator with loss Cell.
        optimizer_g (Optimizer): Optimizer for updating the generator weights.
        optimizer_d (Optimizer): Optimizer for updating the discriminator weights.
        sens (float): The adjust parameter. Default: 1.0
    """

    def __init__(self, generator, discriminator, optimizer_g, optimizer_d, sens: float = 1.0):
        super(TrainOneStepCell, self).__init__(auto_prefix=True)
        self.generator = generator
        self.generator.set_grad()

        self.discriminator = discriminator
        self.discriminator.set_grad()

        self.weights_g = optimizer_g.parameters
        self.optimizer_g = optimizer_g
        self.weights_d = optimizer_d.parameters
        self.optimizer_d = optimizer_d

        self.grad = ops.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens

        # Parallel processing
        self.reducer_flag = False
        self.grad_reducer_g = ops.identity
        self.grad_reducer_d = ops.identity
        self.parallel_mode = _get_parallel_mode()
        self.reducer_flag = self.parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL)
        if self.reducer_flag:
            self.mean = _get_gradients_mean()
            self.degree = _get_device_num()
            self.grad_reducer_g = nn.DistributedGradReducer(self.weights_g, self.mean, self.degree)
            self.grad_reducer_d = nn.DistributedGradReducer(self.weights_d, self.mean, self.degree)

    def construct(self, images, edges, masks):
        '''construct'''
        loss_g = self.generator(images, edges, masks)
        loss_d = self.discriminator(images, edges, masks)
        sens_d = ops.Fill()(ops.DType()(loss_d), ops.Shape()(loss_d), self.sens)
        grads_d = self.grad(self.discriminator, self.weights_d)(images, edges, masks, sens_d)
        res_d = ops.depend(loss_d, self.optimizer_d(grads_d))
        sens_g = ops.Fill()(ops.DType()(loss_g), ops.Shape()(loss_g), self.sens)
        grads_g = self.grad(self.generator, self.weights_g)(images, edges, masks, sens_g)
        res_g = ops.depend(loss_g, self.optimizer_g(grads_g))
        return res_d, res_g


class AdversarialLoss(nn.Cell):
    """
    Adversarial loss.

    Args:
        loss_type (str): The type of loss function(nsgan | lsgan | hinge). Default: 'nsgan'
        target_real_label (float): The value of target real label. Default: 1.0
        target_fake_label (float): The value of target fake label. Default: 0.0
    """

    def __init__(self, loss_type: str = 'nsgan', target_real_label: float = 1.0, target_fake_label: float = 0.0):
        super(AdversarialLoss, self).__init__()

        self.loss_type = loss_type
        self.real_label = mindspore.Parameter(mindspore.Tensor(target_real_label, mindspore.float32),
                                              requires_grad=False)
        self.fake_label = mindspore.Parameter(mindspore.Tensor(target_fake_label, mindspore.float32),
                                              requires_grad=False)
        if self.loss_type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif self.loss_type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif self.loss_type == 'hinge':
            self.criterion = nn.ReLU()

    def construct(self, outputs, is_real, is_disc=None):
        '''construct'''
        if self.loss_type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()

            return (-outputs).mean()
        labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
        loss = self.criterion(outputs, labels)
        return loss


class StyleLoss(nn.Cell):
    """
    Style loss.

    Args:
        vgg_path (str): The path of vgg model checkpoint.
    """

    def __init__(self, vgg_path: str):
        super(StyleLoss, self).__init__()
        self.insert_child_to_cell('vgg', VGG19(vgg_path))
        self.criterion = nn.L1Loss()

    def compute_gram(self, x):
        b, ch, h, w = x.shape
        f = x.view(b, ch, w * h)
        f_t = f.transpose(0, 2, 1)
        gram = P.BatchMatMul()(f, f_t) / (h * w * ch)

        return gram

    def construct(self, x, y):
        '''StyleLoss construct'''
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute loss
        style_loss = 0.0
        style_loss += self.criterion(self.compute_gram(x_vgg['relu2_2']), self.compute_gram(y_vgg['relu2_2']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu3_4']), self.compute_gram(y_vgg['relu3_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu4_4']), self.compute_gram(y_vgg['relu4_4']))
        style_loss += self.criterion(self.compute_gram(x_vgg['relu5_2']), self.compute_gram(y_vgg['relu5_2']))

        return style_loss


class PerceptualLoss(nn.Cell):
    """
    Perceptual loss, VGG-based.

    Args:
        vgg_path (str): The path of vgg model checkpoint.
    """

    def __init__(self, vgg_path: str):
        super(PerceptualLoss, self).__init__()
        self.insert_child_to_cell('vgg', VGG19(vgg_path))
        self.criterion = nn.L1Loss()
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def construct(self, x, y):
        '''PerceptualLoss construct'''
        # Compute features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss


class VGG19(nn.Cell):
    """
    VGG19 model for loss function.

    Args:
        vgg_path (str): The path of vgg model checkpoint.
    """

    def __init__(self, vgg_path: str):
        super(VGG19, self).__init__()
        self.relu1_1 = nn.SequentialCell(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )
        self.relu1_2 = nn.SequentialCell(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )

        self.relu2_1 = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same"),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )
        self.relu2_2 = nn.SequentialCell(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )

        self.relu3_1 = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same"),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )
        self.relu3_2 = nn.SequentialCell(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )
        self.relu3_3 = nn.SequentialCell(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )
        self.relu3_4 = nn.SequentialCell(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )

        self.relu4_1 = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same"),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )
        self.relu4_2 = nn.SequentialCell(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )
        self.relu4_3 = nn.SequentialCell(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )
        self.relu4_4 = nn.SequentialCell(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )

        self.relu5_1 = nn.SequentialCell(
            nn.MaxPool2d(kernel_size=2, stride=2, pad_mode="same"),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )
        self.relu5_2 = nn.SequentialCell(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )
        self.relu5_3 = nn.SequentialCell(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )
        self.relu5_4 = nn.SequentialCell(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3, 3), stride=(1, 1), pad_mode='pad', padding=1,
                      has_bias=True),
            nn.ReLU()
        )

        checkpoint = mindspore.load_checkpoint(vgg_path)
        mindspore.load_param_into_net(self, checkpoint)

        # don't need the gradients, just want the features
        for param in self.get_parameters():
            param.requires_grad = False

    def construct(self, x):
        """VGG forward and get loss function."""
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
        