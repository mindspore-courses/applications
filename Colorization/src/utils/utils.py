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
"""Smooth the animation image and save it in a new directory."""
from __future__ import absolute_import
from __future__ import unicode_literals
import os

import mindspore.ops
import mindspore.nn as nn
import numpy as np
import sklearn.neighbors as sn
from skimage import color


class NNEncLayer:
    """
    Layer which encodes ab map into Q colors

    Args:
        args (namespace): Network parameters.

    Inputs:
        - **img_ab** (tensor) - AB channel of image.

    Outputs:
         - **encode** (ndarray) - Coded image AB channel.
         - **max_encode** (ndarray) - Coding maximum.
    """

    def __init__(self, args):
        self.nne = 32
        self.sigma = 0.5
        self.enc_dir = args.resource
        self.nnenc = NNEncode(self.nne, self.sigma, km_filepath=os.path.join(self.enc_dir, 'pts_in_hull.npy'))

        self.x = 224
        self.y = 224
        self.q = self.nnenc.k

    def forward(self, img_ab):
        encode = self.nnenc.encode_points_mtx_nd(img_ab)
        max_encode = np.argmax(encode, axis=1).astype(np.int32)
        return encode, max_encode

    def reshape(self, top):
        top[0].reshape(self.N, self.Q, self.X, self.Y)


class PriorBoostLayer:
    """
    Layer boosts ab values based on their rarity

    Args:
        args (namespace): Network parameters.

    Inputs:
        - **bottom** (ndarray) - Calculate boost.

    Outputs:
        - **boost** (ndarray) - Layer boosts ab values
    """

    def __init__(self, args):
        self.enc_dir = args.resource
        self.gamma = .5
        self.alpha = 1.
        self.pc = PriorFactor(self.alpha, gamma=self.gamma, priorFile=os.path.join(self.enc_dir, 'prior_probs.npy'))

        self.x = 224
        self.y = 224

    def forward(self, bottom):
        boost = self.pc.forward(bottom, axis=1)
        return boost


class NonGrayMaskLayer:
    """
    Layer outputs a mask based on if the image is grayscale or not

    Inputs:
        - **bottom** (ndarray) - Image ab channel values.

    Outputs:
        - **boost** (ndarray) - Layer outputs a mask
    """

    def forward(self, bottom):
        bottom = bottom.asnumpy()
        mask = (np.sum(np.sum(np.sum((np.abs(bottom) > 5).astype('float'), axis=1),
                              axis=1), axis=1) > 0)[:, na(), na(), na()].astype('float')
        return mask


class PriorFactor:
    """
    Handles prior factor

    Args:
        alpha (int): Prior correction factor.
        gamma (int): Percentage to mix in uniform prior with empirical prior.
        verbose (namespace): Network parameters.
        priorFile(str)： File which contains prior probabilities across classes

    Inputs:
        - **bottom** (ndarray) - Image ab channel values.

    Outputs:
        - **boost** (ndarray) - A priori factor after handles
    """

    def __init__(self, alpha, gamma=0, priorFile=''):
        self.alpha = alpha
        self.gamma = gamma
        # self.verbose = verbose
        self.prior_probs = np.load(priorFile)
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs != 0] = 1.
        self.uni_probs = self.uni_probs / np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution
        self.prior_mix = (1 - self.gamma) * self.prior_probs + self.gamma * self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix ** -self.alpha
        self.prior_factor = self.prior_factor / np.sum(self.prior_probs * self.prior_factor)  # re-normalize

        # implied empirical prior
        self.implied_prior = self.prior_probs * self.prior_factor
        self.implied_prior = self.implied_prior / np.sum(self.implied_prior)  # re-normalize

    def forward(self, data_ab_quant, axis=1):
        """Handles prior factor"""
        data_ab_maxind = np.argmax(data_ab_quant, axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if axis == 0:
            boost = corr_factor[na(), :]
        elif axis == 1:
            boost = corr_factor[:, na(), :]
        elif axis == 2:
            boost = corr_factor[:, :, na(), :]
        elif axis == 3:
            boost = corr_factor[:, :, :, na()]
        return boost



class NNEncode:
    """
    Encode points using NN search and Gaussian kernel

    Args:
        NN (int): Number of nearest neighbors to be selected.
        sigma (int): A uniform distribution with weight.
        km_filepath (str): A prior file path.
        cc(int)：Correction factor

    Inputs:
        - **pts_nd** (ndarray) - Image ab channel values.

    Outputs:
        - **pts_enc_nd** (ndarray) - Coded image AB channel.
    """

    def __init__(self, nne, sigma, km_filepath='', cc=-1):
        self.p_index = None
        self.pts_enc_flt = None
        if check_value(cc, -1):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.k = self.cc.shape[0]
        self.nne = int(nne)
        self.sigma = sigma
        self.nbr = sn.NearestNeighbors(n_neighbors=nne, algorithm='ball_tree').fit(self.cc)
        self.used = False

    def encode_points_mtx_nd(self, pts_nd, axis=1, same_block=True):
        """Encode points using NN search and Gaussian kernel"""
        pts_flt = flatten_nd_array(pts_nd, axis=axis)
        p = pts_flt.shape[0]
        if same_block and self.used:
            self.pts_enc_flt[...] = 0  # already pre-allocated
        else:
            self.used = True
            self.pts_enc_flt = np.zeros((p, self.k))
            self.p_index = np.arange(0, p, dtype='int')[:, na()]
        pts_flt = pts_flt.asnumpy()
        (dists, index) = self.nbr.kneighbors(pts_flt)

        wts = np.exp(-dists ** 2 / (2 * self.sigma ** 2))
        wts = wts / np.sum(wts, axis=1)[:, na()]
        self.pts_enc_flt[self.p_index, index] = wts

        pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)
        return pts_enc_nd


# *****************************
# ***** Utility functions *****
# *****************************
def check_value(cc, val):
    """
    Check to see if an array is a single element equaling a particular value for pre-processing inputs in a function
    """
    if np.array(cc).size == 1:
        if cc == val:
            return True
    return False


def na():
    """shorthand for new axis"""
    return np.newaxis


def flatten_nd_array(pts_nd, axis=1):
    """
    Flatten an nd array into a 2d array with a certain axis

    Inputs:
        - **pts_nd** (ndarray) - Image ab channel values.
        - **axis** (int) - Dimension

    Outputs:
        - **pts_flt**(ndarray)    2d array with a certain axis
    """

    ndim = pts_nd.dim()
    shp = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, ndim), np.array((axis)))  # non axis indices
    npts = np.prod(shp[nax])
    axorder = tuple(np.concatenate((nax, np.array(axis).flatten()), axis=0).tolist())
    transpose = mindspore.ops.Transpose()
    pts_flt = transpose(pts_nd, axorder)
    pts_flt = pts_flt.view(npts.item(), shp[axis].item())
    return pts_flt


def unflatten_2d_array(pts_flt, pts_nd, axis=1, squeeze=False):
    """
    Unflatten a 2d array with a certain axis

    Inputs:
        - **pts_flt** (ndarray) - 2d array with a certain axis
        - **pts_nd** (ndarray) - Image ab channel values.
        - **axis** (int) - Dimension
        - **squeeze** (bool) - whether squeeze it out

    Outputs:
        - **pts_out** (ndarray) - Unflatten a 2d array
    """
    ndim = pts_nd.dim()
    shp = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, ndim), np.array(axis))  # non axis indices

    if squeeze:
        axorder = nax
        axorder_rev = np.argsort(axorder)
        new_shp = shp[nax].tolist()

        pts_out = pts_flt.reshape(new_shp)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
        axorder_rev = tuple(np.argsort(axorder).tolist())
        m = pts_flt.shape[1]
        new_shp = shp[nax].tolist()
        new_shp.append(m)
        pts_out = pts_flt.reshape(new_shp)
        pts_out = pts_out.transpose(axorder_rev)
    return pts_out


def decode(data_l, conv8_313, r_path, rebalance=1):
    """
    Merge gray channel and ab channel to output color images

    Inputs:
        - **data_l** (tensor) - Gray scale channel of image
        - **conv8_313** (tensor) - AB channel of image
        - **rebalance** (int) - Rebalance factor

    Outputs:
        - **img_rgb** (ndarray) - RGB image
    """
    data_l = data_l[0] + 50
    data_l = data_l.asnumpy().transpose((1, 2, 0))
    conv8_313 = conv8_313[0]
    enc_dir = r_path
    conv8_313_rh = conv8_313 * rebalance
    softmax = nn.Softmax(axis=0)
    class8_313_rh = softmax(conv8_313_rh).asnumpy().transpose((1, 2, 0))
    class8 = np.argmax(class8_313_rh, axis=-1)
    cc = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))
    data_ab = cc[class8[:][:]]
    data_ab = data_ab.repeat(4, axis=0).repeat(4, axis=1)
    img_lab = np.concatenate((data_l, data_ab), axis=-1)
    img_rgb = color.lab2rgb(img_lab)
    return img_rgb
