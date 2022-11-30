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
"""DEAL ECG DATA"""

import numpy as np

from src.dataset import data_preprocess as dp


class Mydata:
    """
    A source dataset that reads and parses EU ST-T ECG dataset.

    Args:
        data_path (str): The path of the csv file of data.
        label_path (str): The path of the csv file of label.
        splits (str): Decide of getting training data or test data
    """

    def __init__(self, data_path: str, label_path: str, splits: str):
        np.random.seed(58)
        if splits == "train":
            self.data, self.label, _, _ = dp.preprocess(data_path=data_path, label_path=label_path)
        elif splits == "test":
            _, _, self.data, self.label = dp.preprocess(data_path=data_path, label_path=label_path)

    def __getitem__(self, index):
        return self.data[index, :].reshape((1, 3600)), self.label[index].astype("float32")

    def __len__(self):
        return len(self.data)
