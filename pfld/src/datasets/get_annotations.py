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
# =====================================================================
""" Annotation files obtained from the pts files of the 300 W dataset. """

import os
import numpy as np


def get_kpts(file_path):
    """
    Get all coordinate information.

    Args:
        file_path (str): Path to pts file.

    Returns:
        kpts (list): List of coordinate points.
    """

    kpts = []
    with open(file_path, 'r') as fr:
        ln = fr.readline()
        while not ln.startswith('n_points'):
            ln = fr.readline()

        # skipping the line with '{'
        ln = fr.readline()

        ln = fr.readline()
        while not ln.startswith('}'):
            vals = ln.strip().split(' ')[:2]
            vals = list(map(np.float32, vals))
            kpts.append(vals)
            ln = fr.readline()
    return kpts


def get_Infomation_list(root_dir, info_dir, lines):
    """
    Obtain the annotation file in (coordinate point, face area, attribute, image path) format.

    Args:
        root_dir (str): Catalog of pictures.
        info_dir (str): Catalog of annotations.
        lines (list): Used to store information for each column.
    """

    info_path = os.path.join(root_dir, info_dir)
    files = os.listdir(info_path)
    points_files = [i for i in files if i.endswith('.pts')]
    for index, pf in enumerate(points_files):
        file_path = os.path.join(info_path, pf)
        kpts = get_kpts(file_path)
        if kpts is None:
            continue
        # ibug dir exist a image that has a space in name
        image_file = pf.split('.')[0] + '.jpg'
        image_path = os.path.join(info_path, image_file)
        if not os.path.isfile(image_path):
            image_file = pf.split('.')[0] + '.png'
            image_path = os.path.join(info_path, image_file)
        GT_points = np.asarray(kpts)

        # crop face box
        x_min, y_min = GT_points.min(0)
        x_max, y_max = GT_points.max(0)
        w, h = x_max - x_min, y_max - y_min
        w = h = min(w, h)
        ratio = 0.1
        x_new = x_min - w * ratio
        y_new = y_min - h * ratio
        w_new = w * (1 + 2 * ratio)
        h_new = h * (1 + 2 * ratio)
        x1 = x_new
        x2 = x_new + w_new
        y1 = y_new
        y2 = y_new + h_new

        line = []
        for i in range(68):
            line.append(str(kpts[i][0]))
            line.append(str(kpts[i][1]))
        line.append(str(int(x1)))
        line.append(str(int(y1)))
        line.append(str(int(x2)))
        line.append(str(int(y2)))
        for i in range(6):
            line.append(str(1))
        line.append(os.path.join(info_dir, image_file))
        assert (len(line) == 147)
        lines.append(line)


def get_annotation(root_dir, fw_path_train, fw_path_test):
    """
    Save the annotation files of the training set and test set to the specified path.

    Args:
        root_dir (str): Root path for 300 W dataset.
        fw_path_train (str): The path to save the annotation file of the training dataset.
        fw_path_test (str): The path to save the annotation files for the test dataset.
    """

    train_lines = []
    train_dirs = ['lfpw/trainset', 'ibug', 'afw', 'helen/trainset']
    for train_dir in train_dirs:
        get_Infomation_list(root_dir + '/300W_images', train_dir, train_lines)

    test_lines = []
    test_dirs = ['lfpw/testset', 'helen/testset']
    for test_dir in test_dirs:
        get_Infomation_list(root_dir + '/300W_images', test_dir, test_lines)

    with open(fw_path_train, 'w') as fw:
        for i, line in enumerate(train_lines):
            # print(line)
            for j in range(len(line)):
                fw.write(line[j] + ' ')
            fw.write('\n')

    with open(fw_path_test, 'w') as fw:
        for i, line in enumerate(test_lines):
            for j in range(len(line)):
                fw.write(line[j] + ' ')
            fw.write('\n')
