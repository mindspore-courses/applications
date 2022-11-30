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
Stylegan2 inference with pre-trained network.
"""

import os
import re
import argparse

import mindspore as ms
from mindspore import ops, Tensor, load_checkpoint, load_param_into_net, context
import numpy as np
import PIL.Image

from model.generator import Generator
from train import save_image_grid

os.environ['GLOG_v'] = '3'

def num_range(s):
    """
    Accept either a comma separated list of numbers '1,2,3'
    or a range '1-3' and return as a list of ints.

    Args:
        s (complex): a comma separated list of numbers '1,2,3' or a range '1-3'

    Returns:
        list, a list of ints.

    Examples:
        >>> out = run_range('1, 3, 5, 7')
    """

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def generate(args_infer):
    """
    Generate images using pretrained network ckpt.

    Args:
        args_infer (argparse.Namespace): the parsed args for inference.

    Returns:
        image, generated images.

    Examples:
         >>> generate(args)
    """

    if args_infer.device_target == 'Ascend':
        context.set_context(mode=context.PYNATIVE_MODE)

    context.set_context(device_target=args_infer.device_target)
    context.set_context(device_id=args_infer.device_id)
    ckpt = args_infer.ckpt
    seeds = args_infer.seeds
    truncation_psi = args_infer.truncation_psi
    noise_mode = args_infer.noise_mode
    out_dir = args_infer.out_dir
    img_res = args_infer.img_res
    num_layers = args_infer.num_layers
    grid_size = args_infer.grid_size
    channel_base = 32768 if img_res >= 512 else 16384

    print('Loading networks from "%s"...' % ckpt)
    generator = Generator(z_dim=512, w_dim=512, c_dim=0, img_resolution=img_res, img_channels=3,
                          mapping_kwargs={'num_layers': num_layers},
                          synthesis_kwargs={'channel_base': channel_base,
                                            'channel_max': 512,
                                            'num_fp16_res': 4,
                                            'conv_clamp': 256})
    param_dict = load_checkpoint(ckpt)
    load_param_into_net(generator, param_dict)

    os.makedirs(out_dir, exist_ok=True)

    # Labels.
    label = ms.numpy.zeros([1, generator.c_dim])

    # Generate images.
    imgs = []
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = Tensor(np.random.RandomState(seed).randn(1, generator.z_dim))
        ws = generator.mapping.construct(z, label, truncation_psi=truncation_psi)
        img = generator.synthesis.construct(ws, noise_mode=noise_mode)
        if grid_size is not None:
            imgs.append(img)
        else:
            img = ops.clip_by_value(img.transpose(0, 2, 3, 1) * 127.5 + 128, 0, 255).astype(ms.uint8)
            PIL.Image.fromarray(img[0].asnumpy(), 'RGB').save(f'{out_dir}/seed{seed:04d}.png')
    if grid_size is not None:
        imgs = ops.Concat()(imgs).asnumpy()
        save_image_grid(imgs, os.path.join(out_dir, f'image grid {seeds}.png'),
                        d_range=[-1, 1], size=grid_size)
        print('%dx%d image grid saved!' % (grid_size[0], grid_size[1]))

    print('%d images have been generated!' % (len(seeds)))
    print('Inference completed!')


def parse_args():
    """
    Parameter configuration
    """

    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--device_target', type=str, default='GPU', help='platform')
    parser.add_argument('--device_id', type=int, default=0, help='appoint device_id if more than 1 device exist')
    parser.add_argument('--ckpt', default='./ckpt/ffhq/G_ema.ckpt', help='Network checkpoint')
    parser.add_argument('--seeds', type=num_range, default='66,1518,389,230',
                        help='seeds option is required, input_format=85,265,297,849 or 601-605')
    parser.add_argument('--truncation_psi', type=float, help='Truncation trick', default=0.5)
    parser.add_argument('--num_layers', type=int, help='Number of mapping layers', default=8)
    parser.add_argument('--noise_mode', type=int, help='Noise mode, 0=none, 1=const, 2=random', default=1)
    parser.add_argument('--img_res', type=int, help='Output image resolution, ffhq=1024, lsun_wide=512', default=1024)
    parser.add_argument('--out_dir', type=str, help='Output path', default='./generated_images')
    parser.add_argument('--grid_size', type=num_range, help='two integers a,b, curate images in axb grid')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    generate(parse_args())
