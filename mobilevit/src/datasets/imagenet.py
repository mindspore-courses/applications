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
""" Create the ImageNet2012 dataset from https://image-net.org/. """

import os

import tempfile
from typing import Callable, Optional, Union, Tuple
import numpy as np
from scipy import io

from mindspore._checkparam import Validator

from utils.images import label2index, read_dataset, image_format
from utils.meta import Dataset, ParseDataset
from utils.path import check_dir_exist, check_file_valid, load_json_file, save_json_file
from utils.class_factory import ClassFactory, ModuleType

__all__ = ["ImageNet", "ParseImageNet"]


@ClassFactory.register(ModuleType.DATASET)
class ImageNet(Dataset):
    """
    A source dataset that reads, parses and augments the IMAGENET dataset.

    The generated dataset has two columns :py:obj:`[image, label]`.
    The tensor of column :py:obj:`image` is a matrix of the float32 type.
    The tensor of column :py:obj:`label` is a scalar of the int32 type.

    Args:
        path (str): The root directory of the IMAGENET dataset or inference image.
        split (str): The dataset split, supports "train", "val" or "infer". Default: "train".
        num_parallel_workers (int, optional): The number of subprocess used to fetch the dataset
            in parallel. Default: None.
        transform (callable, optional):A function transform that takes in a image. Default: None.
        target_transform (callable, optional):A function transform that takes in a label. Default: None.
        batch_size (int): The batch size of dataset. Default: 64.
        repeat_num (int): The repeat num of dataset. Default: 1.
        shuffle (bool, optional): Whether to perform shuffle on the dataset. Default: None.
        num_shards (int, optional): The number of shards that the dataset will be divided. Default: None.
        shard_id (int, optional): The shard ID within num_shards. Default: None.
        resize (Union[int, tuple]): The output size of the resized image. If size is an integer, the smaller edge of the
            image will be resized to this value with the same image aspect ratio. If size is a sequence of length 2,
            it should be  (height, width). Default: 224.

    Raises:
        ValueError: If `split` is not 'train', 'test' or 'infer'.

    Examples:
        >>> from mindvision.classification.dataset import ImageNet
        >>> dataset = ImagenNet("./data/imagenet/", "train")
        >>> dataset = dataset.run()

    About IMAGENET dataset:

    IMAGENET is an image dataset that spans 1000 object classes and contains 1,281,167 training images,
    50,000 validation images and 100,000 test images. Images of each object are quality-controlled and
    human-annotated.

    You can unzip the dataset files into this directory structure and read them by MindSpore Vision's API.

    .. code-block::

        .imagenet/
        ├── train/  (1000 directories and 1281167 images)
        │  ├── n04347754/
        │  │   ├── 000001.jpg
        │  │   ├── 000002.jpg
        │  │   └── ....
        │  └── n04347756/
        │      ├── 000001.jpg
        │      ├── 000002.jpg
        │      └── ....
        └── val/   (1000 directories and 50000 images)
        ├── n04347754/
        │   ├── 000001.jpg
        │   ├── 000002.jpg
        │   └── ....
        └── n04347756/
            ├── 000001.jpg
            ├── 000002.jpg
            └── ....

    Citation

    .. code-block::

        @inproceedings{deng2009imagenet,
        title        = {Imagenet: A large-scale hierarchical image database},
        author       = {Deng, Jia and Dong, Wei and Socher, Richard and Li, Li-Jia and Li, Kai and Fei-Fei, Li},
        booktitle    = {2009 IEEE conference on computer vision and pattern recognition},
        pages        = {248--255},
        year         = {2009},
        organization = {IEEE}
        }
    """

    def __init__(self,
                 path: str,
                 split: str = "train",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 batch_size: int = 64,
                 resize: Union[Tuple[int, int], int] = 224,
                 repeat_num: int = 1,
                 shuffle: Optional[bool] = None,
                 num_parallel_workers: int = 1,
                 num_shards: Optional[int] = None,
                 shard_id: Optional[int] = None,
                 ):
        Validator.check_string(split, ["train", "val", "infer"], "split")

        self.images_path, self.images_label = self.__load_data(os.path.join(os.path.expanduser(path), split))
        load_data = read_dataset if split == "infer" else self.read_dataset

        super(ImageNet, self).__init__(path=path,
                                       split=split,
                                       load_data=load_data,
                                       transform=transform,
                                       target_transform=target_transform,
                                       batch_size=batch_size,
                                       repeat_num=repeat_num,
                                       resize=resize,
                                       shuffle=shuffle,
                                       num_parallel_workers=num_parallel_workers,
                                       num_shards=num_shards,
                                       shard_id=shard_id,
                                       )

    @property
    def index2label(self):
        """
        Get the mapping of indexes and labels.
        """
        parse_imagenet = ParseImageNet(self.path)
        if not os.path.exists(os.path.join(parse_imagenet.path, parse_imagenet.meta_file)):
            parse_imagenet.parse_devkit()

        wind2class_name = load_json_file(os.path.join(parse_imagenet.path, parse_imagenet.meta_file))['wnid2class']
        wind2class_name = sorted(wind2class_name.items(), key=lambda x: x[0])
        mapping = {}

        for index, (_, class_name) in enumerate(wind2class_name):
            mapping[index] = class_name[0]

        return mapping

    @staticmethod
    def __load_data(path):
        """Read each image and its corresponding label from directory."""
        check_dir_exist(path)

        available_label = set()
        images_label, images_path = [], []
        label_to_idx = label2index(path)

        # Iterate each file in the path
        for label in label_to_idx.keys():
            for file_name in os.listdir(os.path.join(path, label)):
                if check_file_valid(file_name, image_format):
                    images_path.append(os.path.join(path, label, file_name))
                    images_label.append(label_to_idx[label])
                    if label not in available_label:
                        available_label.add(label)

        empty_label = set(label_to_idx.keys()) - available_label

        if empty_label:
            raise ValueError(f"Found invalid file for the label {','.join(empty_label)}.")

        return images_path, images_label

    def read_dataset(self, *args):
        if not args:
            return self.images_path, self.images_label

        return np.fromfile(self.images_path[args[0]], dtype="int8"), self.images_label[args[0]]


class ParseImageNet(ParseDataset):
    """
    Parse ImageNet dataset and generate the json file (file name:imagenet_meta.json).
    """

    devkit_file = ["meta.mat", "ILSVRC2012_validation_ground_truth.txt"]
    meta_file = "imagenet_meta.json"

    def __parse_meta_mat(self, devkit_path):
        """Parse the mat file(meta.mat)."""
        metafile = os.path.join(devkit_path, "data", self.devkit_file[0])
        meta = io.loadmat(metafile, squeeze_me=True)['synsets']

        nums_children = list(zip(*meta))[4]
        meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]

        idcs, wnids, classes = list(zip(*meta))[:3]
        clssname = [tuple(clss.split(', ')) for clss in classes]
        idx2wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
        wnid2class = {wnid: clss for wnid, clss in zip(wnids, clssname)}
        return idx2wnid, wnid2class

    def __parse_groundtruth(self, devkit_path):
        """Parse ILSVRC2012_validation_ground_truth.txt."""
        val_gt = os.path.join(devkit_path, "data", self.devkit_file[1])
        with open(val_gt, "r") as f:
            val_idx2image = f.readlines()
        return [int(i) for i in val_idx2image]

    def parse_devkit(self):
        """Parse the devkit archive of the ImageNet2012 classification dataset and save meta info in json file."""

        with tempfile.TemporaryDirectory() as temp_dir:
            devkit_path = os.path.join(temp_dir, "ILSVRC2012_devkit_t12")
            idx2wnid, wnid2class = self.__parse_meta_mat(devkit_path)
            val_idcs = self.__parse_groundtruth(devkit_path)
            val_wnids = [idx2wnid[idx] for idx in val_idcs]

            # Generating imagenet_meta.json which saved the values of wnid2class and val_wnids
            dict_json = {"wnid2class": wnid2class, "val_wnids": val_wnids}
            save_json_file(os.path.join(self.path, self.meta_file), dict_json)

    # pylint: disable=unused-argument
    def parse_dataset(self, *args):
        """Parse the devkit archives of ImageNet dataset."""
        if not os.path.exists(os.path.join(self.path, self.meta_file)):
            self.parse_devkit()
