import glob
from importlib import invalidate_caches
from operator import le
import os
from re import S
from typing import List, Tuple
import mindspore as m
import mindspore.dataset as d
from mindspore.dataset.transforms.transforms import PyTensorOperation, TensorOperation
import mindspore.dataset.vision as V
import mindspore.dataset.transforms as T
import numpy as np

from src.dataset_utils import (
    RandomGamma,
    TransformsComposeForMultiImages,
    get_father_dir,
    read_data,
)
from src.config import FLYINGCHAIRS_VALIDATE_INDICES, SINTEL_VALIDATE_INDICES


class FlyingChairsDataset:
    def __init__(
        self,
        root: str,
        augmentations: List[Tuple[PyTensorOperation, TensorOperation]],
        imgtype="final",
        split: str = "train",
    ) -> None:
        image_files = sorted(glob.glob(os.path.join(root, "*.ppm")))
        flow_files = sorted(glob.glob(os.path.join(root, "*.flo")))

        num_flow = len(flow_files)
        validata_indices = [
            x for x in FLYINGCHAIRS_VALIDATE_INDICES if x in range(num_flow)
        ]
        train_indices = [x for x in range(num_flow) if x not in validata_indices]
        print(
            f"FlyingChairsDataset: {len(validata_indices)} val flows, {len(train_indices)} train flows"
        )

        if split == "train":
            canddiate_indices = train_indices
        elif split == "val":
            canddiate_indices = validata_indices
        elif split == "full":
            canddiate_indices = range(num_flow)
        else:
            raise ValueError(f"Invalid split: {split}")

        self.image_path_list = []
        self.flow_path_list = []
        for i in canddiate_indices:
            self.image_path_list.append([image_files[2 * i], image_files[2 * i + 1]])
            self.flow_path_list.append(flow_files[i])

        self.simple_transform = TransformsComposeForMultiImages([V.ToPIL(), V.ToTensor()])
        
        if augmentations:
            self.transform = TransformsComposeForMultiImages(augmentations)
        else:
            self.transform = TransformsComposeForMultiImages([V.ToPIL(), V.ToTensor()])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        index = index % len(self)

        image1 = self.image_path_list[index][0]
        image2 = self.image_path_list[index][1]
        flow = self.flow_path_list[index]

        im1_arr = read_data(image1)
        im2_arr = read_data(image2)
        flow_arr = read_data(flow)

        im1, im2 = self.transform(im1_arr, im2_arr)
        flo = np.transpose(flow_arr, (2, 0, 1))

        return im1, im2, flo


class SintelDataset:
    def __init__(
        self,
        root: str,
        augmentations: List[Tuple[PyTensorOperation, TensorOperation]],
        img_type="final",
        split: str = "train",
    ) -> None:

        self.split = split

        image_root_path = os.path.join(root, img_type)
        flow_root = os.path.join(root, "flow")

        images_path = sorted(glob.glob(os.path.join(image_root_path, "*/*.png")))
        flows_path = sorted(glob.glob(os.path.join(flow_root, "*/*.flo")))

        dir_full_base = get_father_dir(images_path[0])
        class_folders = [
            os.path.dirname(os.path.abspath(fn)).replace(dir_full_base, "") for fn in images_path
        ]
        base_folders = sorted(list(set(class_folders)))

        self.image_item = []
        self.flow_item = []


        for bf in base_folders:
            img_path = [x for x in images_path if bf in x]
            flow_path = [x for x in flows_path if bf in x]

            for i in range(len(img_path) - 1):
                im1 = img_path[i]
                im2 = img_path[i + 1]
                flo = flow_path[i]

                self.image_item.append([im1, im2])
                self.flow_item.append([flo])

        full_num = len(self.image_item)
        validata_indices = [x for x in SINTEL_VALIDATE_INDICES if x in range(full_num)]
        canddiate_indices = None
        train_indices = [x for x in range(full_num) if x not in validata_indices]
        print(
            f"SintelDataset: {len(validata_indices)} val flows, {len(train_indices)} train flows"
        )

        if split == "train":
            canddiate_indices = [
                x for x in range(full_num) if x not in validata_indices
            ]
        elif split == "val":
            canddiate_indices = validata_indices
        elif split == "full":
            canddiate_indices = range(full_num)
        else:
            raise ValueError(f"Invalid split: {split}")

        self.image_item = [self.image_item[i] for i in canddiate_indices]
        self.flow_item = [self.flow_item[i] for i in canddiate_indices]

        self.simple_transform = TransformsComposeForMultiImages([V.ToPIL(), V.ToTensor()])
        
        if augmentations:
            self.transform = TransformsComposeForMultiImages(augmentations)
        else:
            self.transform = TransformsComposeForMultiImages([V.ToPIL(), V.ToTensor()])

    def __len__(self):
        return len(self.image_item)
        # return 1

    def __getitem__(self, index):
        index = index % len(self)

        im1_path = self.image_item[index][0]
        im2_path = self.image_item[index][1]
        flo_path = self.flow_item[index][0]

        im1_arr = read_data(im1_path)
        im2_arr = read_data(im2_path)
        flo_arr = read_data(flo_path)

        if self.split == "train":
            x0 = np.random.randint(0, im1_arr.shape[1] - 512)
            y0 = np.random.randint(0, im1_arr.shape[0] - 384)
            im1_arr = im1_arr[y0 : y0 + 384, x0 : x0 + 512, :]
            im2_arr = im2_arr[y0 : y0 + 384, x0 : x0 + 512, :]
            flo_arr = flo_arr[y0 : y0 + 384, x0 : x0 + 512, :]

        im1, im2 = self.transform(im1_arr, im2_arr)
        flo = np.transpose(flo_arr, (2, 0, 1))

        return im1, im2, flo


def getFlyingChairsTrainData(
    root, augmentations, split, batch_size, num_parallel_workers
):
    dataset = FlyingChairsDataset(root, augmentations, split)
    data_loader = (
        d.GeneratorDataset(
            dataset,
            ["im1", "im2", "flo"],
            shuffle=True,
            num_parallel_workers=num_parallel_workers,
        )
        .map(input_columns="im1", operations=T.TypeCast(m.float32))
        .map(input_columns="im2", operations=T.TypeCast(m.float32))
        .map(input_columns="flo", operations=T.TypeCast(m.float32))
        .batch(batch_size, drop_remainder=True)
    )

    return data_loader, len(dataset), dataset

def getSintelValData(root, augmentations, split, batch_size, num_parallel_workers):
    dataset = SintelDataset(root, augmentations, split=split)
    data_loader = (
        d.GeneratorDataset(
            dataset,
            ["im1", "im2", "flo"],
            shuffle=True if split == "train" else False,
            num_parallel_workers=num_parallel_workers,
        )
        .map(input_columns="im1", operations=T.TypeCast(m.float32))
        .map(input_columns="im2", operations=T.TypeCast(m.float32))
        .map(input_columns="flo", operations=T.TypeCast(m.float32))
        .batch(batch_size, drop_remainder=True)
    )

    return data_loader, len(dataset), dataset


# augmentation_list = [
#             V.ToPIL(),
#             V.RandomColorAdjust(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
#             V.ToTensor(),
#             RandomGamma(min_gamma=0.7, max_gamma=1.5, clip_image=True)
#             ]
# sintel_dataset = SintelDataset(
#     root=r'K:\Action\data\MPI-Sintel-complete\training',
#     augmentations=augmentation_list
# )
# print(len(sintel_dataset))

# flyingchairs_dataset = FlyingChairsDataset(
#     root=r'K:\Action\data\FlyingChairs_release\data',
#     augmentations=augmentation_list
# )
# print(len(flyingchairs_dataset))
