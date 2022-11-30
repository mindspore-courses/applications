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
"""Test CRA model."""

import glob
import os
import time
import argparse
import cv2
import numpy as np
import progressbar
from scipy import signal

import mindspore
from mindspore import nn, ops, Tensor
from mindspore import context, load_checkpoint, load_param_into_net

from models.inpainting_network import GatedGenerator
from models.compute_attention import ApplyAttention2


def sort(str_lst):
    """Return the sorted list in ascending order."""

    return [s for s in sorted(str_lst)]


def read_imgs_masks(args):
    """
    Sort the image and mask directories in order and return it.

    Args:
        args(class): option class.

    Return:
        paths_img: Return to the image list in order.
        paths_mask: Return to the mask list in order.
    """

    paths_img = glob.glob(args.image_dir + '/*.*[g|G]')
    paths_img = sort(paths_img)
    paths_mask = glob.glob(args.mask_dir + '/*.*[g|G]')
    paths_mask = sort(paths_mask)
    return paths_img, paths_mask


def get_input(path_img, path_mask):
    """
    Read and process the image and mask through the given path.

    Args:
        path_img(str): image path.
        path_mask(str): mask path.

    Return:
        image[0]: Can input images of network models.
        mask[0]: Can input masks of network models.
    """

    image = cv2.imread(path_img)
    mask = cv2.imread(path_mask)
    image = np.expand_dims(image, 0)
    mask = np.expand_dims(mask, 0)
    return image[0], mask[0]


def build_inference_net(raw_img_ph, raw_mask_ph, model_gen, args):
    """
    Complete CRA network testing model, including image preprocessing, generator generation and output,
        and image post-processing operations.

    Args:
        raw_img_ph(Tensor): image read from folder.
            It is processed into the format of [1,3,512,512], the data type is float32, and normalized.
        raw_mask_ph(Tensor): mask read from folder.
            It is processed into the format of [1,3,512,512], the data type is float32, and normalized.
        model_gen(cell): generation network.
        args(class): option class.

    Return:
        raw_size_output: Large test output results.
        raw_img_ph: Image read from folder.
        raw_mask_ph: Mask read from folder.
    """

    # Process input image
    raw_img = ops.ExpandDims()(raw_img_ph, 0)
    raw_img = raw_img.astype(mindspore.float32)
    raw_img = ops.Transpose()(raw_img, (0, 3, 1, 2))
    resize = ops.ResizeNearestNeighbor((args.times * args.input_size, args.times * args.input_size))
    large_img = resize(raw_img)
    large_img = ops.Reshape()(large_img, (1, 3, args.times * args.input_size, args.times * args.input_size))
    large_img = large_img / 127.5 - 1
    net = nn.Unfold([1, args.times, args.times, 1], [1, args.times, args.times, 1], [1, 1, 1, 1], 'same')
    small_img = net(large_img)
    small_img = ops.Transpose()(small_img, (0, 2, 3, 1))
    small_img = ops.Reshape()(small_img, (1, args.input_size, args.input_size, args.times, args.times, 3))
    small_img = ops.ReduceMean(False)(small_img, axis=(3, 4))
    small_img = ops.Transpose()(small_img, (0, 3, 1, 2))
    # Process input mask
    raw_mask = ops.ExpandDims()(raw_mask_ph, 0)
    raw_mask = raw_mask.astype(mindspore.float32)
    raw_mask = ops.Transpose()(raw_mask, (0, 3, 1, 2))
    resize = ops.ResizeNearestNeighbor((args.input_size, args.input_size))
    small_mask = resize(raw_mask)
    small_mask = ops.Reshape()(small_mask, (1, 3, args.input_size, args.input_size))
    small_mask = 1 - small_mask / 255
    # Input image and mask to genenrator
    x2, _, corres = build_inference_graph(real=small_img, mask=small_mask, model_gen=model_gen)
    # Post processing
    large_output, _, _, _ = post_processing(large_img, small_img, x2, small_mask, corres, args)
    # Resize back
    raw_size_output = resize_back(raw_img, large_output, small_mask)
    return raw_size_output, raw_img_ph, raw_mask_ph


def build_inference_graph(real, mask, model_gen):
    """
    Input real and mask to generator and output the results.

    Return:
        x2: Generator output.
        fake_patched: The inside of the mask area is filled by the generated results,
             and the outside of the mask area is filled by the original image.
        corres: Attention score.
    """

    mask = mask[0:1, 0:1, :, :]
    x = real * (1. - mask)
    _, x2, corres = model_gen(x, mask)
    fake = x2
    fake_patched = fake * mask + x * (1 - mask)
    return x2, fake_patched, corres


def post_processing(large_img, small_img, low_base, small_mask, corres, args):
    """
     Subtracting the large blurry image from the raw input to compute contextual residuals,
     and calculate aggregated residuals through attention transfer module.
     Adding the aggregated residuals to the up-sampled generator inpainted result.

    Args:
        large_img(Tensor): the input image resize nearestneighbored to [1,3,4096,4096], and normalized.
        small_img(Tensor): the large_img processed into [1,3,512,512].
        low_base(Tensor): generator output.
        small_mask(Tensor): the input mask is processed into [1,3,512,512], and normalized.
        corres(Tensor): Attention score.
        args(class): option class.

    Return:
        x: High frequency residual image passes the attention transfer module,
            and add generator output results pass upsampling operation.
        low_raw: Low-frequency image [1,3,4096,4096].
        low_base: Generator output pass upsampling operation.
        residual: High frequency residual image passes the attention transfer module.
    """

    high_raw = large_img
    low_raw = small_img
    mask = 1 - small_mask
    low_raw = nn.ResizeBilinear()(low_raw, scale_factor=args.times)
    to_shape = list(ops.Shape()(mask))[2:]
    to_shape[0], to_shape[1] = int(to_shape[0] * args.times), int(to_shape[1] * args.times)
    resize = ops.ResizeNearestNeighbor((to_shape[0], to_shape[1]))
    mask = resize(mask)
    residual1 = (high_raw - low_raw) * mask
    residual = ApplyAttention2([1, 3, 4096, 4096], [1, 1024, 32, 32])(residual1, corres)
    low_base = nn.ResizeBilinear()(low_base, scale_factor=args.times)
    x = low_base + residual
    x = x.clip(-1, 1)
    x = (x + 1.) * 127.5
    return x, low_raw, low_base, residual


def gaussian_kernel(size, std):
    """Return a gaussian kernel."""

    k = signal.gaussian(size, std)
    kk = np.matmul(k[:, np.newaxis], [k])
    return kk / np.sum(kk)


def resize_back(raw_img, large_output, small_mask):
    """
    Process the test output result in the format of [1, 3,4096,4096] to the same size as the original input image.

    Args:
        raw_img(Tensor): original input image to be tested in NCHW format.
        large_output(Tensor): test output results in the format of [1,3,4096,4096].
        small_mask(Tensor): mask with format [1,3,512,512].

    Return:
        raw_size_output: Final test output results in the format of NHWC, and data type is mindspore.uint8.
    """

    raw_shp = raw_img.shape
    raw_size_output = nn.ResizeBilinear()(large_output, size=(raw_shp[2], raw_shp[3]))
    raw_size_output = raw_size_output.astype(mindspore.float32)
    gauss_kernel = gaussian_kernel(7, 1.)
    gauss_kernel = Tensor(gauss_kernel)
    gauss_kernel = gauss_kernel.astype(mindspore.float32)
    gauss_kernel = ops.ExpandDims()(gauss_kernel, 2)
    gauss_kernel = ops.ExpandDims()(gauss_kernel, 3)
    a, b, c, d = ops.Shape()(gauss_kernel)
    gauss_kernel = ops.Transpose()(gauss_kernel, (3, 2, 0, 1))
    conv = nn.Conv2d(c, d, (a, b), 1, pad_mode='same', padding=0, weight_init=gauss_kernel, data_format='NCHW')
    mask = conv(small_mask[:, 0:1, :, :])
    mask = nn.ResizeBilinear()(mask, size=(raw_shp[2], raw_shp[3]))
    mask = mask.astype(mindspore.float32)
    raw_size_output = raw_size_output * mask + raw_img * (1 - mask)
    raw_size_output = ops.Transpose()(raw_size_output, (0, 2, 3, 1))
    raw_size_output = raw_size_output.astype(mindspore.uint8)
    return raw_size_output


def parse_args():
    """Parse parameters."""

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, help='The directory of images to be tested.')
    parser.add_argument('--mask_dir', type=str, help='The directory of masks.')
    parser.add_argument('--output_dir', type=str, help='Where to write testing output.')
    parser.add_argument('--checkpoint_dir', type=str, help='The directory of loading checkpoint.')
    parser.add_argument('--attention_type', default='SOFT', type=str, help='compute attention type.')
    parser.add_argument('--train_batchsize', default=1, type=int, help='Batch size for testing.')
    parser.add_argument('--input_size', default=512, type=int, help='The image size of the input network in the test.')
    parser.add_argument('--times', default=8, type=int, help='The scaling size of input image.')
    return parser.parse_args()


if __name__ == "__main__":
    context.set_context(mode=context.PYNATIVE_MODE)

    # setting test data
    cra_config = parse_args()
    img_paths, mask_paths = read_imgs_masks(cra_config)
    if not os.path.exists(cra_config.output_dir):
        os.makedirs(cra_config.output_dir)
    total_time = 0
    bar = progressbar.ProgressBar(maxval=len(img_paths), widgets=[progressbar.Bar('=', '[', ']'), ' ',
                                                                  progressbar.Percentage()])
    bar.start()

    # load net and checkpoint file
    gen = GatedGenerator(cra_config)
    param_dict = load_checkpoint(cra_config.checkpoint_dir)
    load_param_into_net(gen, param_dict)

    # test
    for (i, img_path) in enumerate(img_paths):
        rint = i % len(mask_paths)
        bar.update(i + 1)
        img_test, mask_test = get_input(img_path, mask_paths[rint])
        s = time.time()
        input_img_ph = Tensor(img_test)
        input_mask_ph = Tensor(255 - mask_test)
        outputs, input_img_ph, input_mask_ph = build_inference_net(input_img_ph, input_mask_ph, gen, cra_config)
        res = outputs[0]
        res = res.asnumpy()
        total_time += time.time() - s
        img_hole = img_test * (1 - mask_test / 255) + mask_test
        res = np.concatenate([img_test, img_hole, res], axis=1)
        cv2.imwrite(cra_config.output_dir + '/' + str(i) + '.jpg', res)
        print('test finish')
    bar.finish()
    print('average time per image', total_time / len(img_paths))
