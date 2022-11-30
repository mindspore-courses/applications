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
""" Create the Inshop dataset. """

import os
import zipfile
import json

import PIL.Image
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


class Inshop:
    """
    A source dataset that reads and parses the Inshop dataset.

    The generated dataset has three columns :py:obj:`[image, label, pose]`.
    The tensor of column :py:obj:`image` is of the uint8 type.
    The tensor of column :py:obj:`label` is a scalar of the float32 type.
    The tensor of column :py:obj:`pose` is a scalar of the float32 type.

    Args: 
        path (str): Path to the root directory that contains the dataset.
        batch_size (int): The batch size of the dataset. Default: 32.
        resize (int): The resize value of images. Default: 256.
        pose_file (str): The file of pose. Default: "None".
        use_labels (bool): Whether to use labels. Default: False.
        max_size (int): The max number of dataset. Default: 48674.
        xflip (bool): Whether need to flip the dataset. Default: False.
        random_seed (int): The random seed. Default: 0.

    Raises: 
        RuntimeError: If `path` does not contain data files.

    Examples:
        >>> dataset_dir = "/path/to/inshop_dataset_directory"
        >>> pose_dir = "/path/to/pose_file_directory"
        >>> inshop =  Inshop(path=dataset_dir, pose_file=pose_dir, xflip=True)
        >>> (img, label, pose) = inshop[0]
    """
    def __init__(self, path, batch_size=32, resize=256, pose_file="None", use_labels=False, max_size=48674,
                 xflip=False, random_seed=0):
        self._path = path
        self._zipfile = None
        resolution = resize
        self._use_labels = use_labels
        self.batch_size = batch_size

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path)
                                for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')

        self.df = pd.read_csv(pose_file)
        self.image_size = resolution
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None

        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        """
        Get raw labels.

        Returns:
            numpy.ndarray, the label.

        Examples:
            >>> raw_labels = self._get_raw_labels()
        """
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 1], dtype=np.float32)
        return self._raw_labels

    def __len__(self):
        """ Return the length of the dataset. """
        return self._raw_idx.size

    def __getitem__(self, idx):
        """
        Get the information by id, including image, id, and pose.

        Args:
            idx (int): The number of images.

        Returns:
            numpy.ndarray, the image.
            numpy.ndarray, the id.
            numpy.ndarray, the pose.

        Examples:
            >>> (img, c, pose) = dataset[idx]
        """
        image = self._load_raw_image(self._raw_idx[idx])
        pose = self.get_pose()
        if self._xflip[idx]:
            image = image[:, :, ::-1]
            pose = np.flip(pose, axis=2)
        return image.copy(), self.get_label(idx), pose

    def get_all(self, ii):
        """
        Get the information by batch, including images, idx, and poses.

        Args:
            ii (int): The number of batch.

        Returns:
            numpy.ndarray, the images.
            numpy.ndarray, the ids.
            numpy.ndarray, the poses.

        Examples:
            >>> (whole_real_img, whole_real_c, whole_pose) = self..get_all(ii)
        """
        images = []
        idxs = []
        poses = []
        for idx in range(ii * self.batch_size, ii * self.batch_size + self.batch_size):
            image = self._load_raw_image(self._raw_idx[idx])
            pose = self.get_pose()

            if self._xflip[idx]:
                image = image[:, :, ::-1]
                pose = np.flip(pose, axis=2)
            images.append(image.copy())
            idxs.append(self.get_label(idx))
            poses.append(pose)
        images = np.array(images)
        idxs = np.array(idxs)
        return images, idxs, poses

    def get_label(self, idx):
        """
        Get label.

        Args:
            idx (int): The idx of the dataset.

        Returns:
            numpy.ndarray, the label.

        Examples:
            >>> label = self.get_label(idx)
        """
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    @property
    def image_shape(self):
        """
        Image shape.

        Returns:
            list, the shape of images.

        Examples:
            >>> shape = self.image_shape
        """
        return list(self._raw_shape[1:])

    @property
    def label_shape(self):
        """
        Label shape.

        Returns:
            list, the shape of labels.

        Examples:
            >>> shape = self.label_shape
        """
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def has_labels(self):
        """
        Whether the dataset has labels.

        Returns:
            bool, whether the dataset has labels.

        Examples:
            >>> flag = self.has_labels
        """
        return False

    @staticmethod
    def _file_ext(fname):
        """
        File extension.

        Args:
            fname (str): File name.

        Returns:
            str, the extension of the file in lower case.

        Examples:
            >>> extension = self._file_ext(fname)
        """
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        """
        Get zipfile.

        Returns:
            function, opened zip file.

        Examples:
            >>> file = self._get_zipfile()
        """
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        """
        Open file.

        Args:
            fname (str): File name.

        Returns:
            function, opened file or None.

        Examples:
            >>> file = self._open_file(fname)
        """
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close_zip(self):
        """ Close the zip. """
        try:
            if self._zipfile is not None:
                self._zipfile.close_zip()
        finally:
            self._zipfile = None

    def _load_raw_image(self, raw_idx):
        """
        Load raw labels.

        Args:
            raw_idx (int): The id of the images.

        Returns:
            numpy.ndarray, raw image.

        Examples:
            >>> image = self._load_raw_image()
        """
        fname = self._image_fnames[raw_idx]
        self.fname = fname
        with self._open_file(fname) as f:
            image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        image = image.transpose(2, 0, 1)
        return image

    def _load_raw_labels(self):
        """
        Load raw labels.

        Returns:
            numpy.ndarray, raw labels.

        Examples:
            >>> labels = self._load_raw_labels()
        """
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')]
                  for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

    def get_pose(self):
        """
        Get the pose of the image by id, shape: [17, 64, 64].

        Returns:
            numpy.ndarray, pose heatmap.

        Examples:
            >>> pose = self.get_pose(idx)
        """
        base = os.path.basename(self.fname)
        keypoint = self.df[self.df['name'] == base]['keypoints'].tolist()
        if len(keypoint) > 0:
            keypoint = keypoint[0]
            ptlist = keypoint.split(':')
            ptlist = [float(x) for x in ptlist]
            maps = self.get_heatmap(ptlist)
        else:
            maps = np.zeros((17, 64, 64))
        return maps

    def get_heatmap(self, pose):
        """
        Pose should be a list of length 51, every 3 number for x, y and confidence for each of the 17 keypoints.

        Args:
            pose (list): Pose list.
            image_size (int): Image size.

        Returns:
            numpy.ndarray, pose heatmap.

        Examples:
            >>> maps = self.get_heatmap(pose, image_size)
        """
        stack = []
        for i in range(17):
            x = pose[3 * i]
            y = pose[3 * i + 1]
            c = pose[3 * i + 2]
            ratio = 64.0 / self.image_size
            map_pose = self.get_gaussian_heatmap([x * ratio, y * ratio])
            if c < 0.4:
                map_pose = 0.0 * map_pose
            stack.append(map_pose)
        maps = np.dstack(stack)
        heatmap = np.transpose(maps, (2, 0, 1))
        return heatmap

    def get_gaussian_heatmap(self, bone_pos):
        """
        Calculate the gaussian heatmap of the position.

        Args:
            bone_pos (list): Position.

        Returns:
            numpy.ndarray, gaussian heatmap.

        Examples:
            >>> map_pose = self.get_gaussian_heatmap([x * ratio, y * ratio])
        """
        width = 64
        x, y = np.mgrid[0:width:1, 0:width:1]
        pos = np.dstack((x, y))
        gau = multivariate_normal(mean=list(bone_pos), cov=[
                                  [width * 0.02, 0.0], [0.0, width * 0.02]]).pdf(pos)
        return gau
