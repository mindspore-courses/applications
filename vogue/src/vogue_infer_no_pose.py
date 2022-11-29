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
"""Vogue inference with trained network. (no pose)"""

import os
import re
import argparse

import numpy as np
import PIL.Image
import mindspore as ms
from mindspore import ops, Tensor, load_checkpoint, load_param_into_net

from models.vogue_generator import GeneratorNoPose as Generator

os.environ['GLOG_v'] = '3'


def num_range(s):
    """
    Accept either a comma separated list of numbers 'a,b,c'
    or a range 'a-c' and return as a list of ints.

    Args:
        s (complex): A comma separated list of numbers 'a,b,c' or a range 'a-c'

    Returns:
        list, a list of ints.

    Examples:
        >>> out = run_range('1, 5, 9, 13')
    """
    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]


def generate_style_mix_no_pose(args_infer):
    """
    Generate images using pretrained network pickle (no pose).

    Args:
        args_infer (argparse.Namespace): The args of inference.

    Examples:
        >>> generate_style_mix_no_pose(args)
    """
    ms.set_context(mode=ms.PYNATIVE_MODE)
    ms.set_context(device_id=args_infer.device)
    ckpt = args_infer.ckpt
    rows = args_infer.rows
    cols = args_infer.cols
    col_styles = args_infer.col_styles
    truncation_psi = args_infer.truncation_psi
    noise_mode = 0 if args_infer.noise_mode == 'const' else 1
    out_dir = args_infer.out_dir
    print('Loading no pose networks from "%s"...' % ckpt)
    whole_seeds = list(set(rows + cols))
    generator = Generator(z_dim=512, w_dim=512, c_dim=0, img_resolution=256, img_channels=3,
                          batch_size=1, mapping_kwargs={'num_layers': 2},
                          synthesis_kwargs={'channel_base': 16384,
                                            'channel_max': 512,
                                            'num_fp16_res': 4,
                                            'conv_clamp': 256})
    param_dict = load_checkpoint(ckpt)
    load_param_into_net(generator, param_dict)

    clip_min = Tensor(0, ms.float32)
    clip_max = Tensor(255, ms.float32)

    os.makedirs(out_dir, exist_ok=True)

    print('Generating W vectors...')
    whole_z = np.stack([np.random.RandomState(seed).randn(generator.z_dim) for seed in whole_seeds])
    whole_w = generator.mapping.construct(Tensor(whole_z, ms.float32), None)
    w_avg = generator.mapping.w_avg
    whole_w = w_avg + (whole_w - w_avg) * truncation_psi
    w_dict = {seed: w for seed, w in zip(whole_seeds, list(whole_w))}
    image_dict = dict()
    print('Generating images...')
    for i in range(whole_w.shape[0]):
        image = generator.synthesis.construct(whole_w[i][np.newaxis], noise_mode=noise_mode)
        image = ops.clip_by_value(image.transpose(0, 2, 3, 1) * 127.5 + 128, clip_min, clip_max).astype(ms.uint8)
        image_dict[(whole_seeds[i], whole_seeds[i])] = image.asnumpy()[0]

    print('Generating style-mixed images...')
    for row_seed in rows:
        for col_seed in cols:
            w = w_dict[row_seed].copy()
            w[col_styles] = w_dict[col_seed][col_styles]
            image = generator.synthesis.construct(w[np.newaxis], noise_mode=noise_mode)
            image = ops.clip_by_value(image.transpose(0, 2, 3, 1) * 127.5 + 128, clip_min, clip_max).astype(ms.uint8)
            image_dict[(row_seed, col_seed)] = image.asnumpy()[0]

    print('Saving images...')
    os.makedirs(out_dir, exist_ok=True)
    for (row_seed, col_seed), image in image_dict.items():
        PIL.Image.fromarray(image, 'RGB').save(f'{out_dir}/{row_seed}-{col_seed}.png')

    print('Saving image grid...')
    ww = generator.img_resolution
    hh = generator.img_resolution
    out = PIL.Image.new('RGB', (ww * (len(cols) + 1), hh * (len(rows) + 1)), 'black')
    for row_idx, row_seed in enumerate([0] + rows):
        for col_idx, col_seed in enumerate([0] + cols):
            if row_idx == 0 and col_idx == 0:
                continue
            key = (row_seed, col_seed)
            if row_idx == 0:
                key = (col_seed, col_seed)
            if col_idx == 0:
                key = (row_seed, row_seed)
            out.paste(PIL.Image.fromarray(image_dict[key], 'RGB'), (ww * col_idx, hh * row_idx))
    out.save(f'{out_dir}/grid.png')


def parse_args():
    """Parameter configuration"""
    args = argparse.ArgumentParser()
    args.add_argument('--ckpt', default='./ckpt/G_ema_no_pose.ckpt',
                      help='Network checkpoint')
    args.add_argument('--rows', type=num_range, default='85,100,75,714',
                      help='Random seeds to use for image rows')
    args.add_argument('--cols', type=num_range, default='33,821,1789,195',
                      help='Random seeds to use for image columns')
    args.add_argument('--col-styles', type=num_range, help='Style layer range',
                      default='0-6')
    args.add_argument('--truncation-psi', type=float, help='Truncation psi',
                      default=1)
    args.add_argument('--noise-mode', help='Noise mode', choices=['const', 'random'],
                      default='const')
    args.add_argument('--out-dir', type=str, default='./out_mixing_no_pose')
    args.add_argument('--device', type=int, default=0, help='device_id')
    args = args.parse_args()
    return args


if __name__ == "__main__":
    generate_style_mix_no_pose(parse_args())
