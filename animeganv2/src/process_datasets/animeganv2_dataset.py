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
""" Create the AnimeGAN dataset. """

import os

import cv2
import mindspore.dataset as ds
import numpy as np

from .utils import normalize_input, compute_data_mean


class AnimeGANDataset:
    """
    A source dataset that downloads, reads, parses and augments the AnimeGAN dataset.

    The generated dataset has two columns :py:obj:`[image, anime, anime_gray, smooth_gray]`.
    The tensor of column :py:obj:`image` is a matrix of the float32 type.
    The tensor of column :py:obj:`anime` is a matrix of the float32 type.
    The tensor of column :py:obj:`anime_gray` is a matrix of the float32 type.
    The tensor of column :py:obj:`smooth_gray` is a matrix of the float32 type.

    Inputs:
        args (namespace): Parameters parsed from dataset.yaml.

    Examples:
        >>> dataset = AnimeGANDataset(args)
        >>> dataset = dataset.run()

    About AnimeGAN dataset:

    The AnimeGAN contains 6,656 real landscape images, 3 animation styles: Hayao,
    Shinkai, Paprika, each animation style is generated from randomly cropped
    images from the film.

    Here is the original AnimeGAN dataset structure.
    You can unzip the dataset files into this directory structure and read them by
    MindSpore Vision's API.

    folder structure:

    .. code-block::

    └── data_dir
         ├── train_photo \n
         ├── dataset # e.g. Hayao
                ├── smooth \n
                ├── style \n

    """

    def __init__(self, args):
        self.batch_size = args.batch_size
        self.dataset = ds.GeneratorDataset(DatasetGenerator(args),
                                           ['image', 'anime', 'anime_gray', 'smooth_gray'],
                                           shuffle=True,
                                           num_parallel_workers=args.num_parallel_workers)

    def run(self):
        """Dataset pipeline."""
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)

        return self.dataset


class DatasetGenerator:
    """
    Dataset generator for getting image path and its corresponding label.

    Inputs:
        args (namespace): Parameters parsed from configuration file.
        transform (function): Image conversion functions, such as rotation, cropping, scaling.
    """

    def __init__(self, args, transform=None):
        data_dir = args.data_dir
        dataset = args.dataset

        anime_dir = os.path.join(data_dir, dataset)
        if not os.path.exists(data_dir):
            print(f'Folder {data_dir} does not exist')

        if not os.path.exists(anime_dir):
            print(f'Folder {anime_dir} does not exist')

        self.mean = compute_data_mean(os.path.join(anime_dir, 'style'))
        print(f'Mean(B, G, R) of {dataset} are {self.mean}')

        self.debug_samples = args.debug_samples or 0
        self.image_files = {}
        self.photo = f'{data_dir}/train_photo'
        self.style = f'{anime_dir}/style'
        self.smooth = f'{anime_dir}/smooth'

        for opt in [self.photo, self.style, self.smooth]:
            folder = opt
            files = os.listdir(folder)

            self.image_files[opt] = [os.path.join(folder, fi) for fi in files]

        self.transform = transform

        print(f'Dataset: real {len(self.image_files[self.photo])} style '
              f'{self.len_anime}, smooth {self.len_smooth}')

    def __len__(self):
        return self.debug_samples or len(self.image_files[self.photo])

    @property
    def len_anime(self):
        """Length of anime image dataset."""
        return len(self.image_files[self.style])

    @property
    def len_smooth(self):
        """Length of smoothed anime image dataset."""
        return len(self.image_files[self.smooth])

    def __getitem__(self, index):
        """
        Load all required data by index.

        Args:
            index (int): Data index.

        Returns:
            Ndarray, real world images.
            Ndarray, anime images.
            Ndarray, gray anime images.
            Ndarray, smoothed anime images.
        """

        image = self.load_photo(index)
        anm_idx = index
        if anm_idx > self.len_anime - 1:
            anm_idx -= self.len_anime * (index // self.len_anime)

        anime, anime_gray = self.load_anime(anm_idx)
        smooth_gray = self.load_anime_smooth(anm_idx)

        return image, anime, anime_gray, smooth_gray

    def load_photo(self, index):
        """
        Load and transform real world images.

        Args:
            index (int): Data index.

        Returns:
            Ndarray, real world images
        """

        fpath = self.image_files[self.photo][index]
        image = cv2.imread(fpath)[:, :, ::-1]
        image = self._transform(image, add_mean=False)
        image = image.transpose(2, 0, 1)
        return image

    def load_anime(self, index):
        """
        Load and transform anime images.

        Args:
            index (int): Data index.

        Returns:
            Ndarray, anime images.
        """

        fpath = self.image_files[self.style][index]
        image = cv2.imread(fpath)[:, :, ::-1]

        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
        image_gray = self._transform(image_gray, add_mean=False)
        image_gray = image_gray.transpose(2, 0, 1)

        image = self._transform(image, add_mean=True)
        image = image.transpose(2, 0, 1)

        return image, image_gray

    def load_anime_smooth(self, index):
        """
        Load and transform smoothed anime images.

        Args:
            index (int): Data index.

        Returns:
            Ndarray, smoothed anime images.
        """

        fpath = self.image_files[self.smooth][index]
        image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)
        image = self._transform(image, add_mean=False)
        image = image.transpose(2, 0, 1)

        return image

    def _transform(self, img, add_mean=True):
        """
        Simple image transform, add mean and normalization.

        Args:
            img (ndarray): Input image.
            add_mean (bool): Whether to add an average pixel value. Default: True.

        Returns:
            Ndarray, transformed image.
        """

        if self.transform is not None:
            img = self.transform(image=img)['image']

        img = img.astype(np.float32)
        if add_mean:
            img += self.mean

        return normalize_input(img)
