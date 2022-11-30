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

import os

import wfdb
import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

np.random.seed(7)


def original_process(rootdir: str):
    """
    Preprocessing of EU ST-T ECG original dataset,
    and save data and label to csv file.

    Args:
        rootdir (str): The path of EU ST-T ECG original dataset.
    """

    files = os.listdir(rootdir)  # 列出文件夹下所有
    name_list = []  # name_list=[100,101,...234]
    mliii = []  # 用 MLIII型导联采集的人（根据选择的不同导联方式会有变换）
    type0 = {}  # 标记及其数量

    for file in files:
        if file[0:5] in name_list:  # 选取文件的前五个字符（可以根据数据文件的命名特征进行修改）
            continue
        else:
            name_list.append(file[0:5])

    for name in name_list:  # 遍历每一个人
        if name[1] not in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']:  # 判断——跳过无用的文件: ('ANNOT', 'thank'),余94个
            continue
        record = wfdb.rdrecord(rootdir + '/' + name)  # 读取一条记录（100），不用加扩展名
        if 'MLIII' in record.sig_name:  # 这里我们记录MLIII导联的数据（也可以记录其他的，根据数据库的不同选择数据量多的一类导联方式即可）
            mliii.append(name)  # 记录下这个人
        annotation = wfdb.rdann(rootdir + '/' + name, 'atr')  # 读取一条记录的atr文件，扩展名atr
        for symbol in annotation.symbol:  # 记录下这个人所有的标记类型
            if symbol in list(type0.keys()):
                type0[symbol] += 1
            else:
                type0[symbol] = 1
        print('sympbol_name', type0)

    sorted(type0.items(), key=lambda d: d[1], reverse=True)

    f = 250  # 数据库的原始采样频率
    segmented_len = 10  # 将数据片段裁剪为10s
    count = 0

    segmented_data = []  # 最后数据集中的X
    segmented_label = []  # 最后数据集中的Y

    print('begin!')
    for person in mliii:  # 读取导联方式为MLIII的数据
        k = 0
        whole_signal = wfdb.rdrecord(rootdir + '/' + person).p_signal.transpose()  # 这个人的一整条数据
        while (k + 1) * f * segmented_len <= len(whole_signal[0]):  # 只要不到最后一组数据点
            count += 1
            record = wfdb.rdrecord(rootdir + '/' + person, sampfrom=k * f * segmented_len,
                                   sampto=(k + 1) * f * segmented_len)  # 读取一条记录（100），不用加扩展名
            annotation = wfdb.rdann(rootdir + '/' + person, 'atr', sampfrom=k * f * segmented_len,
                                    sampto=(k + 1) * f * segmented_len)  # 读取一条记录的atr文件，扩展名atr
            lead_index = record.sig_name.index('MLIII')  # 找到MLII导联对应的索引
            signal = record.p_signal.transpose()  # 两个导联，转置之后方便画图
            label = []  # 这一段数据对应的label，最后从这里面选择最终的label
            # segmented_data.append(signal[lead_index])   # 只记录MLII导联的数据段
            symbols = annotation.symbol
            re_signal = scipy.signal.resample(signal[lead_index], 3600)  # 采样
            re_signal_3 = np.round(re_signal, 3)
            print('resignal', re_signal_3)
            segmented_data.append(re_signal_3)
            print('symbols', symbols, len(symbols))
            if not symbols:
                segmented_label.append('Q')
            elif symbols.count('N') / len(symbols) == 1 or symbols.count('N') + symbols.count('/') == len(
                    symbols):  # 如果全是'N'或'/'和'N'的组合，就标记为N
                segmented_label.append('N')
            else:
                for i in symbols:
                    if i != 'N':
                        label.append(i)
                segmented_label.append(label[0])
            k += 1
    print('begin to save dataset!')
    segmented_data = pd.DataFrame(segmented_data)
    segmented_label = pd.DataFrame(segmented_label)
    segmented_data.to_csv('30X_eu_MLIII.csv', index=False)
    segmented_label.to_csv('30Y_eu_MLIII.csv', index=False)
    print('Finished!')


def preprocess(data_path: str, label_path: str):

    """
    ECG data reprocessing.

    Args:
        data_path (str): The path of data.
        label_path (str): The path of label.

    Returns:
       x_train (np.ndarray): The train data.
       y_train (np.ndarray): The train label.
       x_test (np.ndarray): The test data.
       y_test (np.ndarray): The test label.
    """
    # 加载数据
    print("load data...")
    x = np.loadtxt(data_path, delimiter=',', skiprows=1).astype('float32')  # [choose_index]
    y = np.loadtxt(label_path, dtype="str", delimiter=',', skiprows=1)  # [choose_index]
    aami = ['N', 'L', 'R', 'V', 'A', '|', 'B']
    delete_list = []

    for i in range(len(y)):
        if y[i] not in aami:  # 删除不在AAMI中标签的数
            delete_list.append(i)

    x = np.delete(x, delete_list, 0)
    y = np.delete(y, delete_list, 0)

    # 数据标准化：
    print("begin standard scaler...")
    ss = StandardScaler()
    std_data = ss.fit_transform(x)
    x = np.expand_dims(std_data, axis=2)

    # 把标签编码
    le = preprocessing.LabelEncoder()
    le = le.fit(aami)
    y = le.transform(y)
    print("the label before encoding:", le.inverse_transform([0, 1, 2, 3, 4, 5, 6]))

    # 分层抽样
    print("begin StratifiedShuffleSplit...")
    # n_split=1就只有二八分，如果需要交叉验证，把训练和测试的代码放到for循环里面就可以
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, train_size=0.9, random_state=0)
    sss.get_n_splits(x, y)

    for train_index, test_index in sss.split(x, y):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_train = np.eye(7)[y_train]
    return x_train, y_train, x_test, y_test
