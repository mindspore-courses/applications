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
# ==============================================================================
"""Evaluation with the test dataset."""

import os
import argparse

import numpy as np
from tqdm import tqdm
from mindspore import load_checkpoint, load_param_into_net, context

from model.downsampler import DSN
from model.edsr import EDSR
from model.block import Quantization

from car_utils.metric import compute_psnr_ssim, ValidateCell
from process_dataset.dataset import DIV2KHR, build_dataset, Set5Test


def main(args):
    # Set environment
    context.set_context(mode=args.mode, device_target=args.device_target, device_id=args.device_id)
    if args.mode is context.PYNATIVE_MODE:
        context.set_context(mempool_block_size="20GB")

    scale = args.scale
    benchmark = args.benchmark
    kernel_size = 3 * scale + 1

    #build net
    if args.device_target == "Ascend":
        from plug_in.adaptive_gridsampler_ascend.gridsampler import Downsampler
    elif args.device_target == "GPU":
        from plug_in.adaptive_gridsampler.gridsampler import Downsampler
    else:
        raise ValueError("Unsupported platform. Only support Ascend/GPU")

    kernel_generation_net = DSN(k_size=kernel_size, scale=scale)
    downsampler_net = Downsampler(kernel_size)
    upscale_net = EDSR(32, 256, scale=scale)
    quant = Quantization()

    #load checkpoint
    kgn_dict = load_checkpoint(args.kgn_ckpt_name)
    usn_dict = load_checkpoint(args.usn_ckpt_name)

    load_param_into_net(kernel_generation_net, kgn_dict, strict_load=True)
    load_param_into_net(upscale_net, usn_dict, strict_load=True)
    kernel_generation_net.set_train(False)
    upscale_net.set_train(False)
    downsampler_net.set_train(False)
    quant.set_train(False)
    valid_net = ValidateCell(kernel_generation_net, upscale_net, downsampler_net, quant, scale, scale)

    #read data
    if args.target_dataset == "DIV2KHR":
        val_dataloader = build_dataset(DIV2KHR(os.path.join(args.img_dir, "DIV2K"), "valid"), 1, 1, False)
    elif args.target_dataset in ["Set5", "Set14", "BSDS100", "Urban100"]:
        val_dataloader = build_dataset(Set5Test(args.img_dir, args.target_dataset), 1, 1, False)
    else:
        raise ValueError("Unsupported dataset. Only support DIV2KHR/Set5/Set14/BSDS100/Urban100")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    psnr_list = list()
    ssim_list = list()
    save_dir = args.output_dir
    for i, data in enumerate(tqdm(val_dataloader.create_dict_iterator(), total=val_dataloader.get_dataset_size())):
        img = data['image']
        downscaled_img, reconstructed_img = valid_net(img)
        psnr, ssim = compute_psnr_ssim(img, downscaled_img, reconstructed_img, i, save_dir, scale, benchmark)
        psnr_list.append(psnr)
        ssim_list.append(ssim)

    print('Mean PSNR: {0:.2f}'.format(np.mean(psnr_list)))
    print('Mean SSIM: {0:.4f}'.format(np.mean(ssim_list)))


def parse_args():
    """
    parse arguments
    """

    parser = argparse.ArgumentParser(description='Content Adaptive Resampler for Image downscaling')
    parser.add_argument('--device_target', default='Ascend', choices=['GPU', 'Ascend'], type=str)
    parser.add_argument('--device_id', default=3, type=int)
    parser.add_argument('--img_dir', type=str, default='./datasets', help='path to the HR images to be downscaled')
    parser.add_argument('--target_dataset', default='DIV2KHR', type=str)
    parser.add_argument('--scale', default=2, type=int, help='downscale factor')
    parser.add_argument('--output_dir', type=str, default='./exp_res', help='path to store results')
    parser.add_argument('--benchmark', type=bool, default=True, help='report benchmark results')
    parser.add_argument('--mode', type=int, default=context.GRAPH_MODE, help='GRAPH_MODE or PYNATIVE_MODE')
    parser.add_argument('--kgn_ckpt_name', type=str, default='')
    parser.add_argument('--usn_ckpt_name', type=str, default='')
    return parser.parse_args()

if __name__ == "__main__":
    main(parse_args())
