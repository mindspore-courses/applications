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
# =========================================================================
"""
Dataset.
"""

import os
from io import BytesIO

from PIL import Image

import numpy as np

import mindspore.dataset as de
import mindspore.common.dtype as mstype
from mindspore.dataset.vision.utils import Inter
import mindspore.dataset.vision.c_transforms as C
import mindspore.dataset.vision.py_transforms as P
import mindspore.dataset.transforms.c_transforms as C2

from process_datasets.random_erasing import RandomErasing
from process_datasets.auto_augment import rand_augment_transform
from utils.mixup import MixUp


def create_dataset(dataset_path,
                   split="train",
                   image_size=224,
                   interpolation=Inter.BICUBIC,
                   crop_min=0.08,
                   batch_size=32,
                   num_workers=8,
                   auto_augment="rand-m9-mstd0.5-inc1",
                   mix_up=0.0,
                   mix_up_prob=1.0,
                   switch_prob=0.5,
                   cut_mix=1.0,
                   h_flip=0.5,
                   re_prop=0.25,
                   re_mode='pixel',
                   re_count=1,
                   label_smoothing=0.1,
                   num_classes=1000):
    """
    Processing of data sets.

    Args:
        dataset_path (str): Dataset path.
        split (str): Whether is train. Default is train.
        image_size (int): Image size. Default: 224.
        interpolation (Inter): Interpolation type. Default: Inter.BICUBIC.
        crop_min (float): Min crop. Default: 0.08.
        batch_size (int): Batch size. Default: 32.
        num_workers (int): Num worker. Default: 8.
        auto_augment (str): Augment type. Default: "rand-m9-mstd0.5-inc1".
        mix_up (float): Mix up. Default: 0.0.
        mix_up_prob (float): MixUp prob. Default: 1.0.
        switch_prob (float): Switch prob. Default: 0.5.
        cut_mix (float): Cut mix. Default: 0.25.
        h_flip (float): Flip. Default: "flip".
        re_prop (float): Re prop. Default: 0.25.
        re_mode (str): Re mode. Default: "pixel".
        re_count (float): Re count. Default: 1..
        label_smoothing (float): Label smoothing. Default: 0.1.
        num_classes (int): Number of classes. Default: 1000.

    Returns:
        Dataset, The processed data set.
    """

    # Determine the number of devices and process serial numbers.
    device_num = int(os.getenv("RANK_SIZE", '1'))
    rank_id = int(os.getenv('RANK_ID', '0'))

    # The BICUBIC (bi-cubic interpolation) method will be used by default.
    print('The {} method will be used by default.'.format('BICUBIC'))

    if split == "train":
        # ImageFolderDataset reads images from a tree-structured file directory to build the source dataset,
        # and all images in the same folder will be assigned the same label.
        ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_workers, shuffle=True,
                                   num_shards=device_num, shard_id=rank_id)
    else:
        # Determine the number of batches processed per step.
        batch_per_step = batch_size * device_num
        print("Eval batch per step: {}.".format(batch_per_step))
        if batch_per_step < 50000:
            if 50000 % batch_per_step == 0:
                num_padded = 0
            else:
                num_padded = batch_per_step - (50000 % batch_per_step)
        else:
            num_padded = batch_per_step - 50000
        print("Eval dataset num_padded: {}.".format(num_padded))

        if num_padded != 0:
            # Padded with decode.
            write_io = BytesIO()
            Image.new('RGB', (image_size, image_size), (255, 255, 255)).save(write_io, 'JPEG')
            padded_sample = {
                'image': np.array(bytearray(write_io.getvalue()), dtype='uint8'),
                'label': np.array(-1, np.int32)
            }

            # Define sample data.
            sample = [padded_sample for x in range(num_padded)]

            # Construct the dataset from the defined sample data.
            ds_pad = de.PaddedDataset(sample)

            # Read images from a tree-structured file directory to build the source data set.
            ds_image_folder = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_workers)
            ds = ds_pad + ds_image_folder

            # Slice the dataset for distributed training.
            distribute_sampler = de.DistributedSampler(num_shards=device_num, shard_id=rank_id, shuffle=False,
                                                       num_samples=None)
            ds.use_sampler(distribute_sampler)
        else:
            ds = de.ImageFolderDataset(dataset_path, num_parallel_workers=num_workers,
                                       shuffle=False, num_shards=device_num, shard_id=rank_id)

    # Define transform operations.
    if split == "train":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        aa_params = dict(
            translate_const=int(image_size * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        assert auto_augment.startswith('rand')
        aa_params['interpolation'] = interpolation
        trans = [

            # Crop the input image at a random position, decode the cropped image in RGB mode,
            # and adjust the size of the decoded image.
            C.RandomCropDecodeResize(image_size, scale=(crop_min, 1.0), ratio=(3 / 4, 4 / 3),
                                     interpolation=interpolation),

            # Perform a horizontal random flip of the input image with a given probability.
            C.RandomHorizontalFlip(prob=h_flip),
            P.ToPIL()
        ]

        # Rand-Augment Data augment.
        trans += [rand_augment_transform(auto_augment, aa_params)]
        trans += [
            P.ToTensor(),
            P.Normalize(mean=mean, std=std),
            RandomErasing(probability=re_prop, mode=re_mode, max_count=re_count)
        ]

    else:
        mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
        std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
        trans = [
            C.Decode(),
            C.Resize(int(256 / 224 * image_size), interpolation=interpolation),
            C.CenterCrop(image_size),
            C.Normalize(mean=mean, std=std),
            C.HWC2CHW()
        ]

    type_cast_op = C2.TypeCast(mstype.int32)
    ds = ds.map(input_columns="image", num_parallel_workers=num_workers, operations=trans, python_multiprocessing=True)
    ds = ds.map(input_columns="label", num_parallel_workers=num_workers, operations=type_cast_op)

    ds = ds.batch(batch_size, drop_remainder=True)

    # MixUp in a data enhancement method proposed by FAIR in 2017: two different images are randomly and linearly.
    # combined while generating linearly combined labels.

    # A value of the parameter MixUp greater than 0 indicates that Mixup data enhancement is enabled.
    # CutMix simply selects two images from the dataset, and then crops a part of one image and superimposes
    # it on top of the other image as the new input image to the network for training.
    if (mix_up > 0. or cut_mix > 0.) and (split == "train"):
        mix_up_fn = MixUp(
            mix_up_alpha=mix_up, cut_mix_alpha=cut_mix,
            cut_mix_minmax=None, prob=mix_up_prob,
            switch_prob=switch_prob,
            label_smoothing=label_smoothing,
            num_classes=num_classes)

        ds = ds.map(operations=mix_up_fn, input_columns=["image", "label"],
                    num_parallel_workers=num_workers)

    return ds


def get_dataset(args, split="train"):
    """
    Get dataset.

    Args:
        args: Various parameters
        split (str): Whether it is train. Default is train.

    Returns:
        Dataset, The processed data set.
    """
    if split == "train":
        data = create_dataset(dataset_path=args.dataset_path,
                              image_size=args.image_size,
                              interpolation=args.interpolation,
                              auto_augment=args.auto_augment,
                              mix_up=args.mixup,
                              cut_mix=args.cutmix,
                              mix_up_prob=args.mixup_prob,
                              switch_prob=args.switch_prob,
                              re_prop=args.re_prop,
                              re_mode=args.re_mode,
                              re_count=args.re_count,
                              num_classes=args.num_classes,
                              label_smoothing=args.label_smoothing,
                              crop_min=args.crop_min,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
    else:
        data = create_dataset(dataset_path=args.eval_path,
                              split="eval",
                              image_size=args.image_size,
                              num_classes=args.num_classes,
                              interpolation=args.interpolation,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers)
    return data
