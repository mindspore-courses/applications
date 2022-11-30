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
"""convert dataset to zip"""

import functools
import io
import json
import os
import zipfile
from pathlib import Path
import argparse

import numpy as np
import PIL.Image
from tqdm import tqdm


def maybe_min(a, b):
    """
    Return the minimum of a and b if b is not None else a

    Args:
        a (int): Variable a.
        b (int): Variable b.

    Returns:
        int, the smaller one.

    Examples:
        >>>  out = maybe_min(5, 6)
    """
    if b is not None:
        return min(a, b)
    return a


def file_ext(name):
    """
    Return the file suffix.

    Args:
        name (str): Name.

    Returns:
        str, the extension of the name.

    Examples:
        >>> ext = file_ext(fname)
    """
    return str(name).split('.')[-1]


def is_image_ext(fname):
    """
    Check whether the file suffix is an image.

    Args:
        fname (str): Name.

    Returns:
        bool, whether the extension is an image.

    Examples:
        >>> flag = is_image_ext(fname)
    """
    ext = file_ext(fname).lower()
    return f'.{ext}' in PIL.Image.EXTENSION


def open_image_folder(source_dir, *, max_images):
    """
    Open source image folder and return the iteration of images.

    Args:
        source_dir (str): Source image folder.
        max_images (int): Output only up to `max-images` images.

    Returns:
        int, max number of source images.
        object, the iteration of images.

    Examples:
        >>> num_files, input_iter = open_image_folder(source, max_images=max_images)
    """
    input_images = [str(f) for f in sorted(Path(source_dir).rglob('*'))
                    if is_image_ext(f) and os.path.isfile(f)]
    labels = {}
    meta_fname = os.path.join(source_dir, 'dataset.json')
    if os.path.isfile(meta_fname):
        with open(meta_fname, 'r') as file:
            labels = json.load(file)['labels']
            if labels is not None:
                labels = { x[0]: x[1] for x in labels }
            else:
                labels = {}

    max_idx = maybe_min(len(input_images), max_images)

    def iterate_images():
        """ The iterator of images. """
        for idx, fname in enumerate(input_images):
            arch_fname = os.path.relpath(fname, source_dir)
            arch_fname = arch_fname.replace('\\', '/')
            img = np.array(PIL.Image.open(fname))
            yield dict(img=img, label=labels.get(arch_fname))
            if idx >= max_idx-1:
                break
    return max_idx, iterate_images()


def make_transform(transform, output_width, output_height, resize_filter):
    """
    Transform for the dataset.

    Args:
        transform (str): The transform, choices=['center-crop', 'center-crop-wide'].
        output_width (int): Output width.
        output_height (int): Output height.
        resize_filter (str): Filter when resizing the images, choices=['box', 'lanczos'].

    Returns:
        numpy.ndarray, functool of the transform.

    Examples:
        >>> transform_image = make_transform(transform, width, height, resize_filter)
    """
    resample = {'box': PIL.Image.BOX, 'lanczos': PIL.Image.LANCZOS}[resize_filter]

    def scale(width, height, img):
        """
        Scale transform.

        Args: width (int): Output width.
               height (int): Output height.
               img (numpy.ndarray): Image.

        Returns:
            numpy.ndarray, output image.

        Examples:
            >>> image = scale(width, height, image)
        """
        w = img.shape[1]
        h = img.shape[0]
        if width == w and height == h:
            return img
        img = PIL.Image.fromarray(img)
        ww = width if width is not None else w
        hh = height if height is not None else h
        img = img.resize((ww, hh), resample)
        return np.array(img)

    def center_crop(width, height, img):
        """
        Crop transform.

        Args:
            width (int): Output width.
            height (int): Output height.
            img (numpy.ndarray): Image.

        Returns:
            numpy.ndarray, output image.

        Examples:
            >>> image = center_crop(width, height, image)
        """
        crop = np.min(img.shape[:2])
        img = img[(img.shape[0] - crop) // 2: (img.shape[0] + crop) // 2,
              (img.shape[1] - crop) // 2: (img.shape[1] + crop) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        return np.array(img)

    def center_crop_wide(width, height, img):
        """
        Wide crop transform.

        Args:
            width (int): Output width.
            height (int): Output height.
            img (numpy.ndarray): Image.

        Returns:
            numpy.ndarray, output image.

        Examples:
            >>> image = center_crop_wide(width, height, image)
        """
        ch = int(np.round(width * img.shape[0] / img.shape[1]))
        if img.shape[1] < width or ch < height:
            return None

        img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
        img = PIL.Image.fromarray(img, 'RGB')
        img = img.resize((width, height), resample)
        img = np.array(img)

        canvas = np.zeros([width, width, 3], dtype=np.uint8)
        canvas[(width - height) // 2 : (width + height) // 2, :] = img
        return canvas

    if transform is None:
        return functools.partial(scale, output_width, output_height)
    if transform == 'center-crop':
        if (output_width is None) or (output_height is None):
            print('must specify --width and --height when using ' + transform + 'transform')
            return None
        return functools.partial(center_crop, output_width, output_height)
    if transform == 'center-crop-wide':
        if (output_width is None) or (output_height is None):
            print('must specify --width and --height when using ' + transform + ' transform')
            return None
        return functools.partial(center_crop_wide, output_width, output_height)
    return None


def open_dataset(source, max_images):
    """
    Load dataset.

    Args:
        source (str): Source path.
        max_images (int): Output only up to `max-images` images.

    Returns:
        int, max number of source images.
        object, the iteration of images.

    Examples:
        >>> num_files, input_iter = open_dataset(source, max_images=max_images)
    """
    if os.path.isdir(source):
        return open_image_folder(source, max_images=max_images)
    else:
        print(f'Missing input file or directory: {source}')
        return None


def open_dest(dest):
    """
    Make dir for the dest, and return the zip operations.

    Args:
        dest (str): The dest path, need to end with 'zip'.

    Returns:
        str, the root dir.
        function, zip write bytes.
        function, close the zip.

    Examples:
        >>> archive_root_dir, save_bytes, close_dest = open_dest(dest)
    """
    dest_ext = file_ext(dest)

    if dest_ext == 'zip':
        if os.path.dirname(dest) != '':
            os.makedirs(os.path.dirname(dest), exist_ok=True)
        zf = zipfile.ZipFile(file=dest, mode='w', compression=zipfile.ZIP_STORED)

        def zip_write_bytes(fname, data):
            zf.writestr(fname, data)
        return '', zip_write_bytes, zf.close
    else:
        print('--dest folder must end with "zip"')
        return None


def convert_dataset(args_dataset):
    """
    Convert an image dataset into a dataset zip.

    Args:
        args_dataset (argparse.Namespace): Args of converting dataset.

    Examples:
        >>> convert_dataset(args)
    """
    source = args_dataset.source
    dest = args_dataset.dest
    max_images = args_dataset.max_images
    transform = args_dataset.transform
    resize_filter = args_dataset.resize_filter
    width = args_dataset.width
    height = args_dataset.height

    PIL.Image.init()
    if dest == '':
        print('--dest output filename or directory must not be an empty string')
    num_files, input_iter = open_dataset(source, max_images=max_images)
    archive_root_dir, save_bytes, close_dest = open_dest(dest)
    transform_image = make_transform(transform, width, height, resize_filter)
    dataset_attrs = None
    labels = []
    for idx, image in tqdm(enumerate(input_iter), total=num_files):
        idx_str = f'{idx:08d}'
        archive_fname = f'{idx_str[:5]}/img{idx_str}.png'
        img = transform_image(image['img'])
        if img is None:
            continue
        channels = img.shape[2] if img.ndim == 3 else 1
        cur_image_attrs = {
            'width': img.shape[1],
            'height': img.shape[0],
            'channels': channels
        }
        if dataset_attrs is None:
            dataset_attrs = cur_image_attrs
            width = dataset_attrs['width']
            height = dataset_attrs['height']
            if width != height:
                print(f'Image dimensions are required to be square. Got {width}x{height}')
            if dataset_attrs['channels'] not in [1, 3]:
                print('Input images must be stored as RGB or grayscale')
            if width != 2 ** int(np.floor(np.log2(width))):
                print('Image width/height after scale and crop are required to be power-of-two')
        elif dataset_attrs != cur_image_attrs:
            err = [f'dataset {k}/cur image {k}: {dataset_attrs[k]}/{cur_image_attrs[k]}'
                   for k in dataset_attrs.keys()]
            print(f'Image {archive_fname} attributes must be equal.  Got:\n' + '\n'.join(err))

        img = PIL.Image.fromarray(img, { 1: 'L', 3: 'RGB' }[channels])
        image_bits = io.BytesIO()
        img.save(image_bits, format='png', compress_level=0, optimize=False)
        save_bytes(os.path.join(archive_root_dir, archive_fname), image_bits.getbuffer())
        labels.append([archive_fname, image['label']] if image['label'] is not None else None)

    metadata = {'labels': labels if all(x is not None for x in labels) else None}
    save_bytes(os.path.join(archive_root_dir, 'dataset.json'), json.dumps(metadata))
    close_dest()


def parse_args():
    """Parameter configuration"""
    args = argparse.ArgumentParser()
    args.add_argument('--source', help='Directory for input dataset',
                      default='../dataset/inshopclothes/', metavar='PATH')
    args.add_argument('--dest', help='Output archive name for output dataset, must end with "zip"',
                      default='../dataset/data2.zip', metavar='PATH')
    args.add_argument('--max-images', help='Output only up to `max-images` images',
                      type=int, default=None)
    args.add_argument('--resize-filter', help='Filter to use when resizing images',
                      choices=['box', 'lanczos'], default='lanczos')
    args.add_argument('--transform', help='Input crop/resize mode, optional',
                      choices=['center-crop', 'center-crop-wide'])
    args.add_argument('--width', help='Output width, optional', type=int)
    args.add_argument('--height', help='Output height, optional', type=int)
    args = args.parse_args()
    return args


if __name__ == "__main__":
    convert_dataset(parse_args())
