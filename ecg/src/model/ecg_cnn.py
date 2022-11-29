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
""" ECG CNN network """

from mindspore import nn


class ECGCNNNet(nn.Cell):
    """
    ECG network definition.

    Args:
       channel_num (tuple): Where the meaning of the element is the number of input channels per network module.
    """

    def __init__(self,
                 channel_num: tuple = (1, 5, 5, 10, 10, 20, 20)):
        super(ECGCNNNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=channel_num[0], out_channels=channel_num[1], kernel_size=3)
        self.bn1 = nn.BatchNorm2d(num_features=channel_num[1])
        self.relu1 = nn.ReLU()
        self.max_pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=channel_num[2], out_channels=channel_num[3], kernel_size=4)
        self.bn2 = nn.BatchNorm2d(num_features=channel_num[3])
        self.relu2 = nn.ReLU()
        self.max_pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv1d(in_channels=channel_num[4], out_channels=channel_num[5], kernel_size=4)
        self.bn3 = nn.BatchNorm2d(num_features=channel_num[5])
        self.relu3 = nn.ReLU()
        self.max_pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.flat = nn.Flatten()
        self.fc1 = nn.Dense(in_channels=9000, out_channels=30)
        self.fc2 = nn.Dense(in_channels=30, out_channels=20)
        self.fc3 = nn.Dense(in_channels=20, out_channels=7)

    def construct(self, x):
        """construct."""
        x = self.conv1(x)
        x = x.reshape((x.shape[0], x.shape[1], 1, x.shape[2]))
        x = self.bn1(x)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[3]))
        x = self.relu1(x)
        x = self.max_pool1(x)
        x = self.conv2(x)
        x = x.reshape((x.shape[0], x.shape[1], 1, x.shape[2]))
        x = self.bn2(x)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[3]))
        x = self.relu2(x)
        x = self.max_pool2(x)
        x = self.conv3(x)
        x = x.reshape((x.shape[0], x.shape[1], 1, x.shape[2]))
        x = self.bn3(x)
        x = x.reshape((x.shape[0], x.shape[1], x.shape[3]))
        x = self.relu3(x)
        x = self.max_pool3(x)
        x = self.flat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
