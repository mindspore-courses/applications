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
"""Buile and train model."""

import argparse

from mindspore import nn, context, Model
from mindspore.dataset import vision
from mindspore import LossMonitor, TimeMonitor, DynamicLossScaleManager
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import get_rank, init, get_group_size

from model.downsampler import DSN
from model.edsr import EDSR
from model.block import Quantization
from process_dataset.dataset import DIV2KHR, build_dataset
from car_utils.metric import ValidateCell
from car_utils.loss import TvLoss, OffsetLoss
from car_utils.callback import SaveCheckpoint

class NetWithLoss(nn.Cell):
    """
    Cell with loss function.
    Wraps the network with loss function.

    Args:
        net1(Cell): To generate kernel weights and kernel offsets.
        net2(Cell): Upsampling net.
        aux_net1(Cell): Downsamping net
        aux_net2(Cell): Quantization net
        offset(int): The unit length on the HR image corresponding pixel on the downscaled image.
        loss1 (Cell): The loss function used to compute loss.
        loss2 (Cell): The loss function used to compute loss.

    Inputs:
        - **image** (Tensor) - Input images.

    Returns:
        loss (float): Loss after calculation
    """

    def __init__(self, net1, net2, aux_net1, aux_net2, offset, loss1, loss2):
        super(NetWithLoss, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.dsn = aux_net1
        self.quant = aux_net2
        self.offset_unit = offset
        self.tv_loss = loss1
        self.offset_loss = loss2
        self.l1_loss = nn.L1Loss()

    def construct(self, image):
        kernels, offsets_h, offsets_v = self.net1(image)
        downscaled_img = self.dsn(image, kernels, offsets_h, offsets_v, self.offset_unit)
        downscaled_img = self.quant(downscaled_img)
        reconstructed_img = self.net2(downscaled_img)
        loss1 = self.l1_loss(reconstructed_img, image)
        loss2 = self.tv_loss(offsets_h, offsets_v, kernels)
        loss3 = self.offset_loss(offsets_h, offsets_v)

        return loss1 + loss2 + loss3


def main(args):
    # Set environment
    context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target)
    num_shards = None
    shard_id = None
    if args.run_distribute:
        init("nccl")
        context.set_auto_parallel_context(
            parallel_mode=context.ParallelMode.DATA_PARALLEL, gradients_mean=True)
        shard_id = get_rank()
        num_shards = get_group_size()
    else:
        context.set_context(device_id=args.device_id)
    scale = args.scale
    kernel_size = 3 * scale + 1

    # create model
    kernel_generation_net = DSN(k_size=kernel_size, scale=scale)
    upscale_net = EDSR(32, 256, scale=scale)

    if args.device_target == "Ascend":
        from plug_in.adaptive_gridsampler_ascend.gridsampler import Downsampler
    elif args.device_target == "GPU":
        from plug_in.adaptive_gridsampler.gridsampler import Downsampler
    downsampler_net = Downsampler(kernel_size)

    quant = Quantization()

    init_epoch = 0
    if args.resume_model:
        para_gen = load_checkpoint(args.kgn_ckpt_name)
        para_dis = load_checkpoint(args.usn_ckpt_name)
        load_param_into_net(kernel_generation_net, para_gen)
        load_param_into_net(upscale_net, para_dis)
        if "epoch" in para_dis.keys():
            init_epoch = int(para_gen["epoch"])


    network = NetWithLoss(kernel_generation_net,
                          upscale_net,
                          downsampler_net,
                          quant,
                          scale,
                          TvLoss(0.005),
                          OffsetLoss(kernel_size, offsetloss_weight=0.001))


    network.set_train(True)

    # init optimizer
    opt_para = list(kernel_generation_net.trainable_params())+list(upscale_net.trainable_params())

    # create dataset
    train_transform = [vision.RandomCrop(args.resize),
                       vision.RandomHorizontalFlip(),
                       vision.RandomVerticalFlip(),
                       vision.ToTensor()]

    train_dataloader = build_dataset(DIV2KHR(args.image_path, "train"),
                                     batch_size=args.batchsize,
                                     repeat_num=args.repeat_num,
                                     shuffle=True,
                                     num_parallel_workers=args.workers,
                                     transform=train_transform,
                                     num_shards=num_shards,
                                     shard_id=shard_id)
    step_size = train_dataloader.get_dataset_size()
    print('Finish loading train dataset, data_size:{}'.format(step_size))

    num_epochs = args.end_epoch
    total_steps = step_size * num_epochs

    lr = nn.dynamic_lr.piecewise_constant_lr([int(0.2*total_steps), int(0.4*total_steps),
                                              int(0.6*total_steps), int(0.8*total_steps),
                                              total_steps],
                                             [1e-4, 5e-5, 1e-5, 5e-6, 1e-6])
    opt = nn.optim.Adam(opt_para, learning_rate=lr, eps=1e-6)

    if args.device_target == "Ascend":
        scale_factor = 4
        scale_window = 3000
        loss_scaler = DynamicLossScaleManager(scale_factor, scale_window)
        model = Model(network=network, optimizer=opt, amp_level="O2", loss_scale_manager=loss_scaler)
    if args.device_target == "GPU":
        model = Model(network=network, optimizer=opt, amp_level="O0")

    eval_network = ValidateCell(kernel_generation_net, upscale_net, downsampler_net, quant, scale, scale)
    test_transform = [vision.CenterCrop(args.resize), vision.ToTensor()]
    val_dataloader = build_dataset(DIV2KHR(args.image_path, "valid"),
                                   batch_size=1,
                                   repeat_num=1,
                                   shuffle=False,
                                   num_parallel_workers=args.workers,
                                   transform=test_transform)
    cb_savecheckpoint = SaveCheckpoint(eval_network, val_dataloader, scale, args.checkpoint_path, args.eval_proid)
    cb_loss = LossMonitor(1)
    time_monitor = TimeMonitor()
    model.train(args.end_epoch,
                train_dataloader,
                callbacks=[cb_loss, cb_savecheckpoint, time_monitor],
                dataset_sink_mode=False,
                initial_epoch=init_epoch)


def parse_arguments():
    """Get arguments"""
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('-j', '--workers', default=1, type=int)
    parser.add_argument('--device_target', default='Ascend', choices=['GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--end_epoch', default=500, type=int)
    parser.add_argument('--batchsize', default=24, type=int)
    parser.add_argument('--repeat_num', default=8, type=int)
    parser.add_argument('--resize', default=192, type=int)
    parser.add_argument('--scale', default=4, type=int, help='downscale factor')
    parser.add_argument('--eval_proid', default=1, type=int)
    parser.add_argument('--checkpoint_path', default='./checkpoint', type=str)
    parser.add_argument('--image_path', default='./datasets/DIV2K', type=str)
    parser.add_argument('--resume_model', action='store_true')
    parser.add_argument('--kgn_ckpt_name', type=str, default='')
    parser.add_argument('--usn_ckpt_name', type=str, default='')
    parser.add_argument('--run_distribute', type=bool, default=False)
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_arguments())
