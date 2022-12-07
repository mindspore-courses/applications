import os
from typing import Any, List, Tuple
import numpy as np
from mindspore.dataset.transforms.transforms import PyTensorOperation, TensorOperation
from matplotlib.pyplot import imread

def read_flo_as_float32(filename):
    with open(filename, 'rb') as file:
        _ = np.fromfile(file, np.int32, count=1)[0]
        w = np.fromfile(file, np.int32, count=1)[0]
        h = np.fromfile(file, np.int32, count=1)[0]    
        data = np.fromfile(file, np.float32, count=2*h*w)
    data2D = np.resize(data, (h, w, 2))
    return data2D

def read_data(filename):
    if filename.endswith('.flo'):
        return read_flo_as_float32(filename)
    else:
        return (imread(filename) * 255.).astype(np.uint8)

def get_father_dir(path):
    return os.path.abspath(os.path.join(os.path.dirname(path), ".."))

class TransformCompose(object):
    def __init__(
        self, transforms: List[Tuple[PyTensorOperation, TensorOperation, object]]
    ) -> None:
        self.transforms = transforms

    def __call__(self, *args: Any) -> Any:
        args_len = len(args)
        args = list(args)
        for t in self.transforms:
            args = list(map(t, args))
        if len(args) == 1:
            return args[0]
        else:
            return args


class TransformsComposeForMultiImages(object):
    def __init__(
        self, transforms: List[Tuple[PyTensorOperation, TensorOperation, object]]
    ) -> None:
        self.transforms = TransformCompose(transforms)

    def __call__(self, *args: np.ndarray) -> Any:
        args_len = len(args)
        concatenated_array = np.concatenate(args, axis=0)
        transformed_array = self.transforms(concatenated_array)
        return np.split(transformed_array, args_len, axis=1)

class RandomGamma:
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False) -> None:
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.clip_image = clip_image
    
    def __call__(self, im):
        gamma = np.random.uniform(self.min_gamma, self.max_gamma)
        im = im ** gamma
        if self.clip_image:
            im = np.clip(im, 0, 1)
        return im
        