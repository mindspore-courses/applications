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
"""Generator loss and discriminator loss."""

import mindspore
from mindspore import load_checkpoint, load_param_into_net
from mindspore import nn

from .color_loss import ColorLoss
from .gram_loss import GramLoss
from .vgg19 import Vgg


def vgg19(vgg19_path, num_classes=1000):
    """
    Get vgg19 neural network with batch normalization.

    Args:
        vgg19_path (str): Path of pretrained VGG19 model.
        num_classes (int): Class numbers. Default: 1000.

    Returns:
        Cell, instance of vgg19 neural network with batch normalization.

    Examples:
        >>> vgg19(vgg19_path, num_classes=1000)
    """

    net = Vgg([64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512], num_classes=num_classes,
              batch_norm=True)
    param_dict = load_checkpoint(vgg19_path)
    load_param_into_net(net, param_dict)
    net.requires_grad = False
    return net


class GeneratorLoss(nn.Cell):
    r"""
    Connection of AnimeGAN generator network and loss.

    .. math::
        L(G) = \omega_{adv}E_{pi\sim S_{data}(p)}\left\lbrack \left( G\left( p_{i} \right) - 1 \right)^{2}
        \right\rbrack + \omega_{con}L_{con}(G,D) + \omega_{gra}L_{gra}(G,D) + \omega_{col}L_{col}(G,D)

    Args:
        discriminator (Cell): Instance of discriminator.
        generator (Cell): Instance of generator.
        args (namespace): Network parameters.

    Inputs:
        - **img** (tensor) - Real world image.
        - **anime_gray** (tensor) - Gray anime image.

    Outputs:
        - **result** (tensor) - Total generator loss.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from src.models.discriminator import Discriminator
        >>> from src.models.generator import Generator
        >>> from src.train import parse_args
        >>> discriminator = Discriminator()
        >>> generator = Generator()
        >>> args = parse_args()
        >>> net_g_with_criterion = GeneratorLoss(discriminator, generator, args)
    """

    def __init__(self, discriminator, generator, args):
        super(GeneratorLoss, self).__init__(auto_prefix=True)
        self.discriminator = discriminator
        self.generator = generator
        self.content_loss = nn.L1Loss()
        self.gram_loss = GramLoss()
        self.color_loss = ColorLoss()
        self.wadvg = args.wadvg
        self.wadvd = args.wadvd
        self.wcon = args.wcon
        self.wgra = args.wgra
        self.wcol = args.wcol
        self.vgg19 = vgg19(args.vgg19_path)
        self.adv_type = args.gan_loss
        self.bce_loss = nn.BCELoss()
        self.relu = nn.ReLU()
        self.adv_type = args.gan_loss

    def construct(self, img, anime_gray):
        """ build network """
        fake_img = self.generator(img)
        fake_d = self.discriminator(fake_img)
        fake_feat = self.vgg19(fake_img)
        anime_feat = self.vgg19(anime_gray)
        img_feat = self.vgg19(img)
        result = self.wadvg * self.adv_loss_g(fake_d) + \
                 self.wcon * self.content_loss(img_feat, fake_feat) + \
                 self.wgra * self.gram_loss(anime_feat, fake_feat) + \
                 self.wcol * self.color_loss(img, fake_img)
        return result

    def adv_loss_g(self, pred):
        """
        Adversarial loss type.

        Args:
            pred (tensor): Tensor output from discriminator.

        Returns:
            Tensor, generator adversarial loss.
        """

        if self.adv_type == 'hinge':
            return -mindspore.numpy.mean(pred)

        if self.adv_type == 'lsgan':
            return mindspore.numpy.mean(mindspore.numpy.square(pred - 1.0))

        if self.adv_type == 'normal':
            return self.bce_loss(pred, mindspore.numpy.zeros_like(pred))

        return mindspore.numpy.mean(mindspore.numpy.square(pred - 1.0))


class DiscriminatorLoss(nn.Cell):
    r"""
    Connection of AnimeGAN discriminator network and loss.

    .. math::
        L(D) = \omega_{adv}\lbrack E_{ai\sim S_{data}(a)}\left\lbrack \left( D\left( a_{i} \right) - 1 \right)^{2}
        \right\rbrack + E_{pi\sim S_{data}(p)}\left\lbrack \left( D\left( G\left( p_{i} \right) \right) \right)^{2}
        \right\rbrack + E_{xi\sim S_{data}(x)}\left\lbrack \left( D\left( x_{i} \right) \right)^{2} \right\rbrack
        + 0.1E_{yi\sim S_{data}(y)}\lbrack\left( D\left( y_{i} \right) \right)^{2}\rbrack

    Args:
        discriminator (Cell): Instance of discriminator.
        generator (Cell): Instance of generator.
        args (namespace): Network parameters.

    Inputs:
        - **img** (tensor) - Real world image.
        - **anime** (tensor) - Original anime image.
        - **anime_gray** (tensor) - Gray anime image.
        - **anime_smt_gray** (tensor) - Smoothed gray anime image.

    Outputs:
        - **result** (tensor) - Total discriminator loss.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> from src.models.discriminator import Discriminator
        >>> from src.models.generator import Generator
        >>> from src.train import parse_args
        >>> discriminator = Discriminator()
        >>> generator = Generator()
        >>> args = parse_args()
        >>> net_d_with_criterion = DiscriminatorLoss(discriminator, generator, args)
    """

    def __init__(self, discriminator, generator, args):
        nn.Cell.__init__(self, auto_prefix=True)
        self.discriminator = discriminator
        self.generator = generator
        self.content_loss = nn.L1Loss()
        self.gram_loss = nn.L1Loss()
        self.color_loss = ColorLoss()
        self.wadvg = args.wadvg
        self.wadvd = args.wadvd
        self.wcon = args.wcon
        self.wgra = args.wgra
        self.wcol = args.wcol
        self.vgg19 = vgg19(args.vgg19_path)
        self.adv_type = args.gan_loss
        self.bce_loss = nn.BCELoss()
        self.relu = nn.ReLU()

    def construct(self, img, anime, anime_gray, anime_smt_gray):
        """ build network """
        fake_img = self.generator(img)
        fake_d = self.discriminator(fake_img)
        real_anime_d = self.discriminator(anime)
        real_anime_gray_d = self.discriminator(anime_gray)
        real_anime_smt_gray_d = self.discriminator(anime_smt_gray)
        result = self.wadvd * (
            1.7 * self.adv_loss_d_real(real_anime_d) +
            1.7 * self.adv_loss_d_fake(fake_d) +
            1.7 * self.adv_loss_d_fake(real_anime_gray_d) +
            1.0 * self.adv_loss_d_fake(real_anime_smt_gray_d)
        )
        return result

    def adv_loss_d_real(self, pred):
        """
        Adversarial loss type of real anime image.

        Args:
            pred (tensor): Tensor output from discriminator.

        Returns:
            Tensor, discriminator adversarial loss for real anime image.
        """

        if self.adv_type == 'hinge':
            return mindspore.numpy.mean(self.relu(1.0 - pred))

        if self.adv_type == 'lsgan':
            return mindspore.numpy.mean(mindspore.numpy.square(pred - 1.0))

        if self.adv_type == 'normal':
            return self.bce_loss(pred, mindspore.numpy.ones_like(pred))

        return mindspore.numpy.mean(mindspore.numpy.square(pred - 1.0))

    def adv_loss_d_fake(self, pred):
        """
        Adversarial loss type of generated anime image.

        Args:
            pred (tensor): Tensor output from discriminator.

        Returns:
            Tensor, discriminator adversarial loss for generated anime image.
        """

        if self.adv_type == 'hinge':
            return mindspore.numpy.mean(self.relu(1.0 + pred))

        if self.adv_type == 'lsgan':
            return mindspore.numpy.mean(mindspore.numpy.square(pred))

        if self.adv_type == 'normal':
            return self.bce_loss(pred, mindspore.numpy.zeros_like(pred))

        return mindspore.numpy.mean(mindspore.numpy.square(pred))
