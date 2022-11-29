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
"""ICT Upsample infer."""

import os
import argparse

import mindspore
import mindspore.ops.operations as P
from mindspore import context

from models.networks import Generator
from datasets.dataset import load_dataset
from upsample_utils.util import postprocess, imsave, AverageMeter
from upsample_utils.metrics import PSNR


def main(opts):
    context.set_context(mode=context.GRAPH_MODE, device_target=opts.device_target, device_id=opts.device_id)
    generator = Generator()
    generator.set_train(False)
    if os.path.exists(opts.ckpt_path):
        print('Strat loading the model parameters from %s' % (opts.ckpt_path))
        checkpoint = mindspore.load_checkpoint(opts.ckpt_path)
        mindspore.load_param_into_net(generator, checkpoint)
        print('Finished load the model')
    psnr_func = PSNR(255.0)

    test_dataset = load_dataset(image_flist=opts.input, edge_flist=opts.prior, mask_filst=opts.mask,
                                image_size=opts.image_size, prior_size=opts.prior_size, mask_type=opts.mask_type,
                                kmeans=opts.kmeans, condition_num=opts.condition_num,
                                augment=False, training=False)
    test_batch_size = 1
    test_dataset = test_dataset.batch(test_batch_size)

    index = 0
    psnr = AverageMeter()
    mae = AverageMeter()
    for sample in test_dataset.create_dict_iterator():
        name = sample['name'].asnumpy()[0]
        images = sample['images']
        edges = sample['edges']
        masks = sample['masks']
        index += test_batch_size
        outputs = generator(images, edges, masks)
        outputs_merged = (outputs * masks) + (images * (1 - masks))
        psnr.update(psnr_func(postprocess(images), postprocess(outputs_merged)), 1)
        mae.update((P.ReduceSum()(P.Abs()(images - outputs_merged)) / P.ReduceSum()(images)), 1)
        output = postprocess(outputs_merged)[0]
        if not opts.test_only:
            path = os.path.join(opts.save_path,
                                name[:name.find('.')] + "_%d" % (index % opts.condition_num) + name[name.find('.'):])
            imsave(output, path)
    print('PSNR: {}, MAE: {}'.format(psnr.avg, mae.avg))


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=int, default='0')
    parser.add_argument('--device_target', type=str, default='GPU', help='GPU or Ascend')
    parser.add_argument('--ckpt_path', type=str, help='model checkpoints path')
    parser.add_argument('--save_path', type=str, help='the path of save result')
    parser.add_argument('--input', type=str, help='path to the input images directory or an input image')
    parser.add_argument('--mask', type=str, help='path to the masks directory or a mask file')
    parser.add_argument('--prior', type=str, help='path to the edges directory or an edge file')
    parser.add_argument('--kmeans', type=str, default='../kmeans_centers.npy', help='path to the kmeans')
    parser.add_argument('--vgg_path', type=str, default='./VGG19.ckpt', help='path to the VGG')
    parser.add_argument('--mode', type=int, default=2, help='1:train, 2:test')
    parser.add_argument('--mask_type', type=int, default=3)
    parser.add_argument('--image_size', type=int, default=256, help='the size of origin image')
    parser.add_argument('--prior_size', type=int, default=32, help='the size of prior image from transformer')
    parser.add_argument('--test_only', action='store_true', help='not save result in test')
    parser.add_argument('--condition_num', type=int, default=1, help='Use how many BERT output')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
