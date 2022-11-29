# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.lr_schedule.py
# org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
""" RecoNet train script."""

import argparse
import os
import time

import mindspore
import mindspore.nn as nn
from mindspore import context

from model.loss import ReCoNetWithLoss
from model.reconet import ReCoNet
from model.vgg import vgg16
from dataset.dataset import load_dataset
from utils.reconet_utils import vgg_encode_image, occlusion_mask_from_flow


def main(args_opt):
    """RecoNet train."""
    train_start = time.time()
    train_mode = context.GRAPH_MODE
    context.set_context(mode=train_mode, device_target=args_opt.device_target)
    # Load Monkaa dataset
    train_dataset = load_dataset(args_opt.monkaa, args_opt.flyingthings3d)
    step_size = train_dataset.get_dataset_size()
    print('dataset size is {}'.format(step_size))

    # Create model.
    reconet = ReCoNet()
    vgg_net = vgg16(args_opt.vgg_ckpt, train_mode)
    style_gram_matrices = vgg_encode_image(vgg_net, args_opt.style_file, args_opt.device_target)

    model = ReCoNetWithLoss(reconet,
                            vgg_net,
                            args_opt.device_target,
                            args_opt.alpha,
                            args_opt.beta,
                            args_opt.gamma,
                            args_opt.lambda_f,
                            args_opt.lambda_o)

    # adam optimizer
    optim = nn.Adam(reconet.trainable_params(), learning_rate=args_opt.learning_rate, weight_decay=0.0)

    train_net = nn.TrainOneStepCell(model, optim)

    global_step = 1
    epochs = args_opt.epochs

    # train by steps
    for epoch in range(epochs):
        for sample in train_dataset.create_dict_iterator():
            occlusion_mask = occlusion_mask_from_flow(
                sample["optical_flow"],
                sample["reverse_optical_flow"],
                sample["motion_boundaries"])

            start = time.perf_counter()
            loss = train_net(sample['frame'], sample['pre_frame'], style_gram_matrices, occlusion_mask,
                             sample["reverse_optical_flow"])

            end = time.perf_counter()
            print(f"Epoch: [{epoch} / {epochs}], "
                  f"step: [{global_step} / {step_size * epochs - 1}], "
                  f"loss: {loss}, "
                  f"time: {(end - start) * 1000: .3f} ms")
            global_step += 1

    # save trained model
    mindspore.save_checkpoint(reconet, os.path.join(args_opt.ckpt_dir, args_opt.output_ckpt))
    train_end = time.time()
    print(f'train finished in {train_end - train_start: .3f} s')

def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description='ReCoNet train.')
    parser.add_argument('--device_target', type=str, default="GPU", choices=["Ascend", "GPU", "CPU"])
    parser.add_argument('--vgg_ckpt', type=str, default=None, help='Path of the vgg16 check point file.')
    parser.add_argument('--style_file', required=True, default=None, help='Location of image.')
    parser.add_argument('--monkaa', type=str, default=None, help='Path of the monkaa dataset.')
    parser.add_argument('--flyingthings3d', type=str, default=None, help='Path of the flyingthings3d dataset.')
    parser.add_argument('--ckpt_dir', type=str, default='./', help='Path to save output ckpt.')
    parser.add_argument('--output_ckpt', type=str, default='reconet.ckpt', help='Saved name for ckpt file.')
    parser.add_argument("--epochs", type=int, default=2, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Value of learning rate.')
    parser.add_argument("--alpha", type=float, default=1e4, help="Weight of content loss")
    parser.add_argument("--beta", type=float, default=1e5, help="Weight of style loss")
    parser.add_argument("--gamma", type=float, default=1e-5, help="Weight of total variation")
    parser.add_argument("--lambda_f", type=float, default=1e5, help="Weight of feature temporal loss")
    parser.add_argument("--lambda_o", type=float, default=2e5, help="Weight of output temporal loss")

    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    main(parse_args())
