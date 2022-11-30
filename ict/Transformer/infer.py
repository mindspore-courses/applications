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
ICT Transformer infer.
"""

import os
import argparse
from PIL import Image

import numpy as np
import mindspore
import mindspore.ops.operations as P
from mindspore import context
from mindspore.train import Model

from models.networks import GPT
from transformer_utils.util import sample_mask


def main(opts):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=opts.device_id)
    kmeans = np.load('../kmeans_centers.npy')
    kmeans = np.rint(127.5 * (kmeans + 1.0))
    kmeans = mindspore.Tensor.from_numpy(kmeans)

    # Define the model and load checkpoint
    block_size = opts.prior_size * opts.prior_size
    transformer = GPT(vocab_size=kmeans.shape[0], n_embd=opts.n_embd, n_layer=opts.n_layer, n_head=opts.n_head,
                      block_size=block_size, use_gelu2=opts.GELU_2, embd_pdrop=0.0, resid_pdrop=0.0, attn_pdrop=0.0)
    if os.path.exists(opts.ckpt_path):
        print('Strat loading the model parameters from %s' % (opts.ckpt_path))
        checkpoint = mindspore.load_checkpoint(opts.ckpt_path)
        mindspore.load_param_into_net(transformer, checkpoint)
        print('Finished load the model')
    transformer.set_train(False)
    model = Model(transformer)

    img_list = []
    for root, _, files in os.walk(opts.image_url):
        for name in files:
            img_list.append(os.path.join(root, name))
    img_list = sorted(img_list)

    mask_list = sorted(os.listdir(opts.mask_url))
    condition_num = opts.condition_num
    for x_url, y_name in zip(img_list, mask_list):
        x_name = x_url.split('/')[-1]

        # load image
        print(x_name)
        x = Image.open(x_url).convert("RGB")
        x = x.resize((opts.prior_size, opts.prior_size), resample=Image.BILINEAR)
        x = mindspore.Tensor.from_numpy(np.array(x)).view(-1, 3)
        x = P.Cast()(x, mindspore.float32)
        x = ((x[:, None, :] - kmeans[None, :, :]) ** 2).sum(-1).argmin(1)

        # load mask
        mask_url = os.path.join(opts.mask_url, y_name)
        y = Image.open(mask_url).convert("L")
        y = y.resize((opts.prior_size, opts.prior_size), resample=Image.NEAREST)
        y = (np.array(y) / 255.) > 0.5
        y = mindspore.Tensor.from_numpy(y).view(-1)
        y = P.Cast()(y, mindspore.float32)

        x_list = [x] * condition_num
        x_tensor = P.Stack()(x_list)
        y_list = [y] * condition_num
        y_tensor = P.Stack()(y_list)
        x_tensor = P.Cast()(x_tensor * (1 - y_tensor), mindspore.int32)
        outputs = sample_mask(model, x_tensor, y_tensor, length=opts.prior_size * opts.prior_size,
                              top_k=opts.top_k)

        # save image
        img_name = x_name[:x_name.find('.')] + x_name[x_name.find('.'):]
        for i in range(condition_num):
            current_url = os.path.join(opts.save_url, 'condition_%d' % (i + 1))
            os.makedirs(current_url, exist_ok=True)
            current_img = kmeans[outputs[i]].view(opts.prior_size, opts.prior_size, 3).asnumpy().astype(np.uint8)
            tmp = Image.fromarray(current_img)
            tmp.save(os.path.join(current_url, img_name))
        print("Finish %s" % (img_name))


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default='0')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt', help='The path of resume ckpt')
    parser.add_argument('--image_url', type=str, default='', help='the folder of image')
    parser.add_argument('--mask_url', type=str, default='', help='the folder of mask')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--prior_size', type=int, default=32, help='input sequence length: prior_size*prior_size')
    parser.add_argument('--save_url', type=str, default='./', help='save the output results')
    parser.add_argument('--condition_num', type=int, default=1, help='Use how many BERT output')
    parser.add_argument('--use_ImageFolder', action='store_true', help='Using the original folder for ImageNet dataset')

    # transformer parameters
    parser.add_argument('--n_layer', type=int, default=35)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--n_embd', type=int, default=1024)
    parser.add_argument('--GELU_2', action='store_true', help='use the new activation function')

    args = parser.parse_known_args()[0]
    return args


if __name__ == "__main__":
    main(parse_args())
