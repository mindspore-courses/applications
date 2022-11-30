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
""" Sceneflow dataset"""
import os

from abc import abstractmethod
from PIL import Image
import mindspore.dataset as ds

from utils.sceneflow_utils import read_flow, future_optical_flow_path, \
    past_optical_flow_path, motion_boundaries_path
from utils import custom_transforms

COLUMNS = ['frame', 'pre_frame', 'optical_flow', 'reverse_optical_flow', 'motion_boundaries']


class SceneflowDataset:
    """
    Sceneflow is the base class for sceneflow dataset. egs. Monkaa, flyingthings3d
    More dataset can be found at https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html

    Args:
        path: str. File path after processing (data enhancement).
    """
    def __init__(self, path: str):
        self.path = path
        self.data = self.load_data()
        self.trans = self.transforms()

    def __getitem__(self, index):
        entry = self.data[index]
        data = {
            'frame': Image.open(entry['frame']).convert("RGB"),
            'pre_frame': Image.open(entry['pre_frame']).convert("RGB"),
            'optical_flow': read_flow(entry['optical_flow']).copy(),
            'reverse_optical_flow': read_flow(entry['reverse_optical_flow']).copy(),
            'motion_boundaries': Image.open(entry['motion_boundaries'])
        }
        for operation in self.trans:
            data = operation(data)
        return (
            data['frame'],
            data['pre_frame'],
            data['optical_flow'],
            data['reverse_optical_flow'],
            data['motion_boundaries']
        )

    @abstractmethod
    def load_data(self):
        """Load dataset"""

    @abstractmethod
    def transforms(self):
        """Transform operation for dataset"""

    def __len__(self):
        return len(self.data)


class Monkaa(SceneflowDataset):
    """
    A source dataset that downloads, reads, parses and augments the Monkaa dataset.

    The generated dataset has five columns :py:obj:`['frame', 'pre_frame', 'optical_flow', 'reverse_optical_flow',
    'motion_boundaries']`.
    The tensor of column :py:obj:`frame` is a matrix of the float32 type.
    The tensor of column :py:obj:`pre_frame` is a matrix of the float32 type.
    The tensor of column :py:obj:`optical_flow` is a matrix of the float32 type.
    The tensor of column :py:obj:`reverse_optical_flow` is a matrix of the float32 type.
    The tensor of column :py:obj:`motion_boundaries` is a matrix of the Bool type.

    Args:
        path (str): The root directory of the monkaa dataset.
        download (bool) : Whether to download the dataset. Default: False.

    About Monkaa dataset:

    The collection contains more than 39000 stereo frames in 960x540 pixel resolution, rendered from various
    synthetic sequences. For details on the characteristics and differences of the three subsets, we refer the reader
    to our paper. The following kinds of data are currently available:

        Segmentations: Object-level and material-level segmentation images.

        Optical flow maps: The optical flow describes how pixels move between images (here, between time steps in a
        sequence). It is the projected screenspace component of full scene flow, and used in many computer vision
        applications.

        Disparity maps: Disparity here describes how pixels move between the two views of a stereo frame. It is a
        formulation of depth which is independent of camera intrinsics (although it depends on the configuration of
        the stereo rig), and can be seen as a special case of optical flow.

        Disparity change maps: Disparity alone is only valid for a single stereo frame. In image sequences,
        pixel disparities change with time. This disparity change data fills the gaps in scene flow that occur when
        one uses only optical flow and static disparity.

        Motion boundaries: Motion boundaries divide an image into regions with significantly different motion. They
        can be used to better judge the performance of an algorithm at discontinuities.

        Camera data: Full intrinsic and extrinsic camera data is available for each view of every stereo frame in our
        dataset collection.

    .. code-block::

        .
        └── Monkaa
             ├── frames_finalpass
             ├── motion_boundaries
             ├── optical_flow

    Citation:

    .. code-block::

        @InProceedings{MIFDB16, author    = "N. Mayer and E. Ilg and P. H{\"a}usser and P. Fischer and D. Cremers and
        A. Dosovitskiy and T. Brox", title     = "A Large Dataset to Train Convolutional Networks for Disparity,
        Optical Flow, and Scene Flow Estimation", booktitle = "IEEE International Conference on Computer Vision and
        Pattern Recognition (CVPR)", year      = "2016", note      = "arXiv:1512.02134", url       =
        "http://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16" }
    """

    def load_data(self):
        """Load monkaa data from path"""
        data_list = []
        for dir_path, _, filenames in os.walk(os.path.join(self.path, "frames_finalpass")):
            length = len(filenames)
            if filenames == '' or length == 0:
                continue

            scene, side = dir_path.split(os.sep)[-2:]

            if side != 'left':
                continue

            filenames.sort()
            filenames = [filename for filename in filenames if filename.endswith(".png")]

            for i in range(1, len(filenames)):
                frame_number = os.path.splitext(filenames[i])[0]
                pre_frame_number = os.path.splitext(filenames[i - 1])[0]
                data_list.append(
                    {
                        'frame': os.path.join(dir_path, filenames[i]),
                        'pre_frame': os.path.join(dir_path, filenames[i - 1]),
                        'optical_flow': future_optical_flow_path(self.path, scene, side, pre_frame_number),
                        'reverse_optical_flow': past_optical_flow_path(self.path, scene, side, frame_number),
                        'motion_boundaries': motion_boundaries_path(self.path, scene, side, frame_number)
                    }
                )
        return data_list

    def transforms(self):
        """Transform operations for monkaa"""
        return [
            custom_transforms.Resize(640, 360),
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.ToTensor()
        ]


class Flyingthings3d(SceneflowDataset):
    """
    A source dataset that downloads, reads, parses and augments the Flyingthings3d dataset.

    The generated dataset has five columns :py:obj:`['frame', 'pre_frame', 'optical_flow', 'reverse_optical_flow',
    'motion_boundaries']`.
    The tensor of column :py:obj:`frame` is a matrix of the float32 type.
    The tensor of column :py:obj:`pre_frame` is a matrix of the float32 type.
    The tensor of column :py:obj:`optical_flow` is a matrix of the float32 type.
    The tensor of column :py:obj:`reverse_optical_flow` is a matrix of the float32 type.
    The tensor of column :py:obj:`motion_boundaries` is a matrix of the Bool type.

    Args:
        path (str): The root directory of the monkaa dataset.

    About Flyingthings3d dataset:

    The collection contains more than 39000 stereo frames in 960x540 pixel resolution, rendered from various
    synthetic sequences. For details on the characteristics and differences of the three subsets, we refer the reader
    to our paper. The following kinds of data are currently available:

        Segmentations: Object-level and material-level segmentation images.

        Optical flow maps: The optical flow describes how pixels move between images (here, between time steps in a
        sequence). It is the projected screenspace component of full scene flow, and used in many computer vision
        applications.

        Disparity maps: Disparity here describes how pixels move between the two views of a stereo frame. It is a
        formulation of depth which is independent of camera intrinsics (although it depends on the configuration of
        the stereo rig), and can be seen as a special case of optical flow.

        Disparity change maps: Disparity alone is only valid for a single stereo frame. In image sequences,
        pixel disparities change with time. This disparity change data fills the gaps in scene flow that occur when
        one uses only optical flow and static disparity.

        Motion boundaries: Motion boundaries divide an image into regions with significantly different motion. They
        can be used to better judge the performance of an algorithm at discontinuities.

        Camera data: Full intrinsic and extrinsic camera data is available for each view of every stereo frame in our
        dataset collection.

    .. code-block::

        .
        └── Flyingthings3d
             ├── frames_finalpass
             ├── motion_boundaries
             ├── optical_flow

    Citation:

    .. code-block::

        @InProceedings{MIFDB16, author    = "N. Mayer and E. Ilg and P. H{\"a}usser and P. Fischer and D. Cremers and
        A. Dosovitskiy and T. Brox", title     = "A Large Dataset to Train Convolutional Networks for Disparity,
        Optical Flow, and Scene Flow Estimation", booktitle = "IEEE International Conference on Computer Vision and
        Pattern Recognition (CVPR)", year      = "2016", note      = "arXiv:1512.02134", url       =
        "http://lmb.informatik.uni-freiburg.de/Publications/2016/MIFDB16" }
    """

    def load_data(self):
        """Load Flyingthings3d data from path"""
        data_list = []
        for dir_path, _, filenames in os.walk(os.path.join(self.path, "frames_finalpass")):
            length = len(filenames)
            if filenames == '' or length == 0:
                continue

            part, subset, scene, side = dir_path.split(os.sep)[-4:]

            if side != 'left':
                continue

            filenames.sort()
            filenames = [filename for filename in filenames if filename.endswith(".png")]

            scene_path = os.path.join(part, subset, scene)

            for i in range(1, len(filenames)):
                frame_number = os.path.splitext(filenames[i])[0]
                pre_frame_number = os.path.splitext(filenames[i - 1])[0]
                data_list.append(
                    {
                        'frame': os.path.join(dir_path, filenames[i]),
                        'pre_frame': os.path.join(dir_path, filenames[i - 1]),
                        'optical_flow': future_optical_flow_path(self.path, scene_path, side, pre_frame_number),
                        'reverse_optical_flow': past_optical_flow_path(self.path, scene_path, side, frame_number),
                        'motion_boundaries': motion_boundaries_path(self.path, scene_path, side, frame_number)
                    }
                )
        return data_list

    def transforms(self):
        """Transform operations for Flyingthings3d"""
        return [
            custom_transforms.Resize(640, 360),
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.ToTensor()
        ]


def load_dataset(monkaa, ft3d):
    """
    Load dataset for ReCoNet

    Args:
        monkaa (str): Path of Monkaa dataset directory
        ft3d (str): Path of Flyingthings3d dataset directory

    Returns:
        dataset with batch_size=2
    """
    monkaa_dataset = ds.GeneratorDataset(Monkaa(monkaa), COLUMNS)
    ft3d_dataset = ds.GeneratorDataset(Flyingthings3d(ft3d), COLUMNS)

    train_dataset = monkaa_dataset + ft3d_dataset
    train_dataset = train_dataset.batch(batch_size=2, drop_remainder=True)
    return train_dataset
