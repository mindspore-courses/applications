import os
import argparse
import numpy as np
from mindspore.mindrecord import FileWriter

seg_schema = {"file_name": {"type": "string"}, "label": {"type": "bytes"}, "data": {"type": "bytes"}}


def build_data(data_root, data_lst, dst_path, num_shards, shuffle):
    data_list = []
    with open(data_lst) as f:
        lines = f.readlines()
    if shuffle:
        np.random.shuffle(lines)

    dst_dir = '/'.join(dst_path.split('/')[:-1])
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    print('number of samples:', len(lines))
    writer = FileWriter(file_name=dst_path, shard_num=num_shards)
    writer.add_schema(seg_schema, "seg_schema")
    cnt = 0

    for l in lines:
        img_path = l.split(' ')[0].strip('\n')
        label_path = l.split(' ')[1].strip('\n')

        sample_ = {"file_name": img_path.split('/')[-1]}
        # print(img_path)
        with open(os.path.join(data_root, img_path), 'rb') as f: #! 去掉绝对路径标识
            sample_['data'] = f.read()
        with open(os.path.join(data_root, label_path), 'rb') as f: #!
            sample_['label'] = f.read()
        data_list.append(sample_)
        cnt += 1
        if cnt % 1000 == 0:
            writer.write_raw_data(data_list)
            print('number of samples written:', cnt)
            data_list = []

    if data_list:
        writer.write_raw_data(data_list)
    writer.commit()
    print('number of samples written:', cnt)
