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
# ======================================================================
""" verification """

import os
import numpy as np
from scipy import signal
from PIL import Image
from skimage.color import rgb2ycbcr

from mindspore import ops, nn

def matlab_style_gauss2d(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma]).
    Acknowledgement:
    https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python (Author@ali_m).

    Args:
        shape(tuple) : The kernel shape. Default: (3, 3).
        sigma(float) : The standard variance. Default: 0.5.

    Returns:
        gauss_f(ndarray) : Gaussian filter.
    """

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    gauss_f = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    gauss_f[gauss_f < np.finfo(gauss_f.dtype).eps * gauss_f.max()] = 0
    sumh = gauss_f.sum()
    if sumh != 0:
        gauss_f /= sumh
    return gauss_f


def calc_ssim(image_1, image_2):
    """
    Please follow the setting of psnr_ssim.m in EDSR.
    (Enhanced Deep Residual Networks for Single Image Super-Resolution CVPRW2017).
    Official Link : https://github.com/LimBee/NTIRE2017/tree/db34606c2844e89317aac8728a2de562ef1f8aba
    The authors of EDSR use MATLAB's ssim as the evaluation tool,
    thus this function is the same as ssim.m in MATLAB with C(3) == C(2)/2.

    Args:
        image_1(ndarray) : Y channel (i.e., luminance) of transformed YCbCr space of X.
        image_2(ndarray) : Y channel (i.e., luminance) of transformed YCbCr space of Y.

    Returns:
        ssim_map : float. The ssim metric.
    """

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    sigma = 1.5
    gaussian_filter = matlab_style_gauss2d((11, 11), sigma)

    image_1 = image_1.astype(np.float64)
    image_2 = image_2.astype(np.float64)

    window = gaussian_filter

    ux = signal.convolve2d(image_1, window, mode='same', boundary='symm')
    uy = signal.convolve2d(image_2, window, mode='same', boundary='symm')

    uxx = signal.convolve2d(image_1 * image_1, window, mode='same', boundary='symm')
    uyy = signal.convolve2d(image_2 * image_2, window, mode='same', boundary='symm')
    uxy = signal.convolve2d(image_1 * image_2, window, mode='same', boundary='symm')

    vx = uxx - ux * ux
    vy = uyy - uy * uy
    vxy = uxy - ux * uy

    ssim_map = ((2 * ux * uy + c1)*(2 * vxy + c2))/((ux ** 2 + uy ** 2 + c1)*(vx + vy + c2))

    return ssim_map.mean()


def cal_psnr(image_true, image_test, benchmark=False):
    """
    Compute the peak signal to noise ratio (PSNR) for an image.

    Args:
        image_true (ndarray): Ground-truth image, same shape as im_test.
        image_test (ndarray): Test image.
        benchmark(bool): report benchmark results. Default: False

    Returns:
        psnr : float. The PSNR metric.
    """

    image_true = np.float64(image_true)
    image_test = np.float64(image_test)

    diff = (image_true - image_test) / 255.0
    if benchmark:
        gray_coeff = np.array([65.738, 129.057, 25.064]).reshape([1, 1, 3]) / 255.0
        diff = diff * gray_coeff
        diff = diff[:, :, 0] + diff[:, :, 1] + diff[:, :, 2]

    mse = np.mean(diff ** 2)
    psnr = -10.0 * np.log10(mse)

    return psnr


def compute_psnr_ssim(image, downscaled_img, reconstructed_img, name, save_dir, scale, benchmark):
    """
    Calculate the evaluation metrics for image.

    Args:
        image (Tensor): Original image
        downscaled_img (Tensor): Downscaled image
        reconstructed_img (Tensor): Reconstructed image
        name (str): The file's name
        save_dir (str): The path to store results
        scale(int): Downscale factor
        benchmark(bool): Weather report benchmark results.

    Returns:
        pnsr: float. The PSNR metric.
        ssim: float. The ssim metric.
    """

    image = image * 255
    image = image.asnumpy().transpose(0, 2, 3, 1)
    image = np.uint8(image)
    orig_img = image[0, ...].squeeze()

    reconstructed_img = ops.clip_by_value(reconstructed_img, 0, 1) * 255
    reconstructed_img = reconstructed_img.asnumpy().transpose(0, 2, 3, 1)
    reconstructed_img = np.uint8(reconstructed_img)
    recon_img = reconstructed_img[0, ...].squeeze()

    downscaled_img = downscaled_img.asnumpy().transpose(0, 2, 3, 1)
    downscaled_img = np.uint8(downscaled_img * 255)
    downscaled_img = downscaled_img[0, ...].squeeze()

    psnr = cal_psnr(orig_img[scale:-scale, scale:-scale, ...],
                    recon_img[scale:-scale, scale:-scale, ...],
                    benchmark=benchmark)

    orig_img_y = rgb2ycbcr(orig_img)[:, :, 0]
    recon_img_y = rgb2ycbcr(recon_img)[:, :, 0]
    orig_img_y = orig_img_y[scale:-scale, scale:-scale, ...]
    recon_img_y = recon_img_y[scale:-scale, scale:-scale, ...]

    ssim = calc_ssim(recon_img_y, orig_img_y)

    if os.path.exists(save_dir):
        image = Image.fromarray(orig_img)
        image.save(os.path.join(save_dir, f'{name}_orig.png'))

        image = Image.fromarray(recon_img)
        image.save(os.path.join(save_dir, f'{name}_recon.png'))

        image = Image.fromarray(downscaled_img)
        image.save(os.path.join(save_dir, f'{name}_down.png'))
    return psnr, ssim


class ValidateCell(nn.Cell):
    """
    Build the validate cell.

    Args:
        net1(Cell): To generate kernel weights and kernel offsets.
        net2(Cell): Upsampling net.
        aux_net1(Cell): Downsamping net
        aux_net2(Cell): Quantization net
        scale(int): Downscale factor
        offset(int): The unit length on the HR image corresponding pixel on the downscaled image.

    Inputs:
        - **img** (Tensor) - The input tensor.

    Outputs:
        Tensor, the downscaled image.
        Tensor, the reconstructed image.
    """

    def __init__(self, net1, net2, aux_net1, aux_net2, scale, offset):
        super(ValidateCell, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.dsn = aux_net1
        self.quant = aux_net2
        self.offset_unit = offset
        self.scale = scale

    def construct(self, image):
        kernels, offsets_h, offsets_v = self.net1(image)
        downscaled_img = self.dsn(image, kernels, offsets_h, offsets_v, self.offset_unit)
        downscaled_img = self.quant(downscaled_img)
        reconstructed_img = self.net2(downscaled_img)

        return downscaled_img, reconstructed_img
