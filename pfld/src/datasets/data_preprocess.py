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
# ==============================================================================
"""Preprocess Method"""

import os
import shutil

from .get_annotations import get_annotation
from .augmentation import get_dataset_list


def data_preprocess(target_dataset: str = '300W',
                    dataset_file_path: str = '../datasets/300W'):
    """
    Used for preprocess dataset, create txt file to help train and evaluate model.

    Args:
        target_dataset: Dataset type.
        dataset_file_path: Dataset path.

    Examples:
        >>> data_preprocess(target_dataset='300W')
    """

    assert target_dataset in ['300W', 'WFLW']
    if target_dataset == '300W':
        # get path of dataset
        root_300w_dir = os.path.dirname(
            os.path.abspath('../') + '/datasets/300W/')

        # define the path to store augmentation result
        fw_path_train = os.path.join(
            root_300w_dir,
            '300W_annotations/list_68pt_rect_attr_train.txt')
        fw_path_test = os.path.join(
            root_300w_dir,
            '300W_annotations/list_68pt_rect_attr_test.txt')

        # store pts file's information into txt file
        get_annotation(root_300w_dir, fw_path_train, fw_path_test)

        # define the label file path and image path
        image_dirs = os.path.join(root_300w_dir, '300W_images')
        mirror_file = os.path.join(
            root_300w_dir, '300W_annotations/Mirror68.txt')
        landmark_dirs = [
            os.path.join(
                root_300w_dir,
                '300W_annotations/list_68pt_rect_attr_train.txt'),
            os.path.join(
                root_300w_dir,
                '300W_annotations/list_68pt_rect_attr_test.txt')]

        train_outdir = dataset_file_path + '/train_data'
        test_outdir = dataset_file_path + '/test_data'
        out_dirs = [train_outdir, test_outdir]

    else:
        root_wflw_dir = os.path.dirname(
            os.path.abspath('../') + '/datasets/WFLW/')

        # define the label file path and image path
        image_dirs = os.path.join(root_wflw_dir, 'WFLW_images')
        mirror_file = os.path.join(
            root_wflw_dir, 'WFLW_annotations/Mirror98.txt')
        landmark_dirs = [
            os.path.join(
                root_wflw_dir,
                'WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_train.txt'),
            os.path.join(
                root_wflw_dir,
                'WFLW_annotations/list_98pt_rect_attr_train_test/list_98pt_rect_attr_test.txt')]

        train_outdir = dataset_file_path + '/train_data'
        test_outdir = dataset_file_path + '/test_data'
        out_dirs = [train_outdir, test_outdir]

    # Do augmentation
    for landmark_dir, out_dir in zip(landmark_dirs, out_dirs):
        # out_dir = os.path.join(root_dir, out_dir)
        print(out_dir)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        if 'list_68pt_rect_attr_test.txt' in landmark_dir or 'list_98pt_rect_attr_test.txt' in landmark_dir:
            is_train = False
        else:
            is_train = True

        get_dataset_list(image_dirs,
                         out_dir,
                         landmark_dir,
                         is_train,
                         target_dataset,
                         mirror_file)
    print('end')
