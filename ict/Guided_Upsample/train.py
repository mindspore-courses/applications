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
"""
ICT Upsample train.
"""

import os
import argparse

import mindspore
import mindspore.nn as nn
import mindspore.ops.operations as P
import mindspore.context as context

from models.networks import Generator, Discriminator
from models.loss import GeneratorWithLoss, DiscriminatorWithLoss, TrainOneStepCell
from datasets.dataset import load_dataset
from upsample_utils.util import postprocess, AverageMeter
from upsample_utils.metrics import PSNR


def main(opts):
    context.set_context(mode=context.GRAPH_MODE, device_target=opts.device_target, device_id=opts.device_id)
    # Define the dataset
    train_dataset = load_dataset(image_flist=opts.input, edge_flist=opts.prior, mask_filst=opts.mask,
                                 image_size=opts.image_size, prior_size=opts.prior_size, mask_type=opts.mask_type,
                                 kmeans=opts.kmeans, use_degradation_2=opts.use_degradation_2,
                                 prior_random_degree=opts.prior_random_degree,
                                 augment=True, training=True)
    train_dataset = train_dataset.batch(opts.batch_size)
    step_size = train_dataset.get_dataset_size()

    # Define the model
    generator = Generator()
    discriminator = Discriminator(in_channels=3)
    psnr_func = PSNR(255.0)
    gen_path = os.path.join(opts.ckpt_path, 'InpaintingModel_gen.ckpt')
    dis_path = os.path.join(opts.ckpt_path, 'InpaintingModel_dis.ckpt')
    if os.path.exists(gen_path):
        print('Strat loading the generator model parameters from %s' % (gen_path))
        checkpoint = mindspore.load_checkpoint(gen_path)
        mindspore.load_param_into_net(generator, checkpoint)
        print('Finished load the model')
    if os.path.exists(dis_path):
        print('Strat loading the discriminator model parameters from %s' % (dis_path))
        checkpoint = mindspore.load_checkpoint(dis_path)
        mindspore.load_param_into_net(discriminator, checkpoint)
        print('Finished load the model')
    model_g = GeneratorWithLoss(generator, discriminator, opts.vgg_path, opts.inpaint_adv_loss_weight,
                                opts.l1_loss_weight, opts.content_loss_weight, opts.style_loss_weight)
    model_d = DiscriminatorWithLoss(generator, discriminator)

    # Define the optimizer
    optimizer_g = nn.Adam(generator.trainable_params(), learning_rate=opts.lr, beta1=opts.beta1, beta2=opts.beta2)
    optimizer_d = nn.Adam(discriminator.trainable_params(), learning_rate=opts.lr * opts.D2G_lr, beta1=opts.beta1,
                          beta2=opts.beta2)

    train_net = TrainOneStepCell(model_g, model_d, optimizer_g, optimizer_d)
    if not os.path.exists(opts.save_path):
        os.mkdir(opts.save_path)
    epoch = 0
    iteration = 0
    keep_training = True
    psnr = AverageMeter()
    mae = AverageMeter()
    psnr_best = 0.0
    if not os.path.exists(opts.save_path):
        os.mkdir(opts.save_path)
    while keep_training:
        epoch += 1
        for i, sample in enumerate(train_dataset.create_dict_iterator()):
            images = sample['images']
            edges = sample['edges']
            masks = sample['masks']
            train_net(images, edges, masks)
            iteration += 1
            if i % 100 == 0:
                outputs = generator(images, edges, masks)
                outputs_merged = (outputs * masks) + (images * (1 - masks))
                psnr.update(psnr_func(postprocess(images), postprocess(outputs_merged)), 1)
                mae.update((P.ReduceSum()(P.Abs()(images - outputs_merged)) / P.ReduceSum()(images)), 1)
                print(f"Epoch: [{epoch}], "
                      f"step: [{i} / {step_size}], "
                      f"psnr: {psnr.avg}, mae: {mae.avg}")
                if psnr.avg > psnr_best:
                    psnr_best = psnr.avg
                    mindspore.save_checkpoint(generator, os.path.join(opts.save_path, 'InpaintingModel_gen_best.ckpt'))
                    mindspore.save_checkpoint(discriminator,
                                              os.path.join(opts.save_path, 'InpaintingModel_dis_best.ckpt'))
            if iteration >= opts.max_iteration:
                keep_training = False
                break
    mindspore.save_checkpoint(generator, os.path.join(opts.save_path, 'InpaintingModel_gen_latest.ckpt'))
    mindspore.save_checkpoint(discriminator, os.path.join(opts.save_path, 'InpaintingModel_dis_latest.ckpt'))


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default='0')
    parser.add_argument('--device_target', type=str, default='GPU', help='GPU or Ascend')
    parser.add_argument('--ckpt_path', type=str, default='', help='model checkpoints path')
    parser.add_argument('--save_path', type=str, default='./checkpoint', help='save checkpoints path')
    parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
    parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
    parser.add_argument('--kmeans', type=str, default='../kmeans_centers.npy', help='path to the kmeans')
    parser.add_argument('--vgg_path', type=str, default='../ckpts_ICT/VGG19.ckpt', help='path to the VGG')
    parser.add_argument('--prior', type=str, default='', help='path to the edges directory or an edge file')
    parser.add_argument('--image_size', type=int, default=256, help='the size of origin image')
    parser.add_argument('--prior_size', type=int, default=32, help='the size of prior image from transformer')
    parser.add_argument('--prior_random_degree', type=int, default=1, help='during training, how far deviate from')
    parser.add_argument('--use_degradation_2', action='store_true', help='use the new degradation function')
    parser.add_argument('--mode', type=int, default=1, help='1:train, 2:test')
    parser.add_argument('--mask_type', default=2, type=int)
    parser.add_argument('--max_iteration', type=int, default=25000, help='How many run iteration')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument("--beta1", type=float, default=0.9, help="Value of beta1")
    parser.add_argument("--beta2", type=float, default=0.9, help="Value of beta2")
    parser.add_argument('--lr', type=float, default=0.0001, help='Value of learning rate.')
    parser.add_argument('--D2G_lr', type=float, default=0.1,
                        help='Value of discriminator/generator learning rate ratio')
    parser.add_argument("--l1_loss_weight", type=float, default=1.0)
    parser.add_argument("--style_loss_weight", type=float, default=250.0)
    parser.add_argument("--content_loss_weight", type=float, default=0.1)
    parser.add_argument("--inpaint_adv_loss_weight", type=float, default=0.1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
