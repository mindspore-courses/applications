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
""" Create the color dataset. """
import os

import numpy as np
from skimage.color import rgb2lab
from PIL import Image
import mindspore
from mindspore import Tensor
from mindspore.dataset import vision as dvision


class ColorizationDataset:
    """
    A source dataset that downloads, reads, parses and augments the ImageNet dataset.

    The generated dataset has two columns :py:obj:`[img_original, img_ab]`.
    The tensor of column :py:obj:`img_original` is a matrix of the float32 type.
    The tensor of column :py:obj:`img_ab` is a matrix of the float32 type.

    Inputs:
        - **path** (str): - Dataset path.
        - **batch_size** (int) - Batch size
        - **shuffle** (bool) - Shuffle dataset
        - **num_parallel_workers** (int) - Specifies the number of worker processes or threads that read data
        - **prob** (float) - Probability of image being flipped

    Examples:
        >>>dataset = ColorizationDataset(args.image_dir, args.batch_size, args.shuffle, args.num_parallel_workers)
        >>>dataset = dataset.run().create_tuple_iterator()

    About AnimeGAN dataset:

    Imagenet contains more than 1.2 million natural images and 1000 categories.The validation set contains
    50000 pictures, and the test set contains 100000 test pictures.

    Here is the original AnimeGAN dataset structure.
    You can unzip the dataset files into this directory structure and read them by
    MindSpore Vision's API.

    folder structure:

    .. code-block::

    └── data_dir
         ├── train \n
                ├── folder1 \n
                ├── folder2 \n

    """
    def __init__(self, path,
                 batch_size,
                 shuffle=True,
                 num_parallel_workers=1,
                 prob=0.5):
        self.batch_size = batch_size
        self.dataset = mindspore.dataset.GeneratorDataset(DatasetGenerator(path, prob), ['img_original', 'img_ab'],
                                                          num_parallel_workers=num_parallel_workers, shuffle=shuffle)

    def run(self):
        """Dataset pipeline."""
        self.dataset = self.dataset.batch(self.batch_size, drop_remainder=True)

        return self.dataset


class DatasetGenerator:
    """
    Dataset generator for getting image path。

    Inputs:
        root_path (string): Directory with all the images.
        transform (callable, optional): Optional transform to be appliedon a sample.
    """

    def __init__(self, root_path, prob):
        self.root = root_path
        dir_list = os.listdir(root_path)
        self.f_list = []
        self.prob = prob
        for i in dir_list:
            files = os.listdir(os.path.join(self.root, i))
            for file in files:
                self.f_list.append(os.path.join(self.root, i, file))

    def __len__(self):
        print(len(self.f_list))
        return len(self.f_list)

    def __getitem__(self, idx):
        img_path = self.f_list[idx]
        img = Image.open(img_path).convert('RGB')
        resize = dvision.Resize(256)
        random_crop = dvision.RandomCrop(224)
        random_crop1 = dvision.Resize(56)
        random_horizontal_flip = dvision.RandomHorizontalFlip(self.prob)
        img_original = random_horizontal_flip(random_crop(resize(img)))
        img_resize = random_crop1(img_original)
        img_original = np.asarray(img_original)
        img_lab = rgb2lab(img_resize)
        img_ab = img_lab[:, :, 1:3]
        img_ab = img_ab.transpose((2, 0, 1))
        img_ab = mindspore.Tensor(img_ab, dtype=mindspore.float32)
        img_original = rgb2lab(img_original)
        img_original = img_original[:, :, 0]-50
        img_original = Tensor(img_original, dtype=mindspore.float32)
        return img_original, img_ab
