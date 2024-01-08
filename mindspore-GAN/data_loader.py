import os
import struct
import random
import numpy as np
import matplotlib.pyplot as plt
from mindspore.dataset import transforms, vision
from mindspore.common import dtype as mstype
from mindspore.dataset import MnistDataset
from download_data import download_MNIST

def preprocess_dataset(dataset, batch_size=64):
    composed_operations = transforms.Compose(
        [
            vision.ToType(mstype.float32),
            vision.Rescale(1.0 / 255.0, 0),
            vision.HWC2CHW(),
            vision.Normalize(mean=(0.5,), std=(0.5,)),
        ]
    )
    dataset = dataset.map(composed_operations, "image")
    # 划分batch
    dataset = dataset.batch(batch_size, True)
    return dataset

def get_train_dataset(batch_size):
    if "train" not in os.listdir("MNIST_Data") or "test" not in os.listdir("MNIST_Data"):
        print(os.listdir("MNIST_Data"))
        download_MNIST()
    train_dataset = MnistDataset('MNIST_Data/train', 'train', shuffle=True)
    train_dataset = preprocess_dataset(train_dataset, batch_size)
    return train_dataset

import matplotlib.pyplot as plt
src_data = './result/src_data.png'

# 可视化部分训练数据
def visualize(dataset):
    sample_iter = dataset.create_dict_iterator(output_numpy=True)
    sample = next(sample_iter)

    figure = plt.figure(figsize=(5, 5))
    cols, rows = 5, 5

    for idx in range(1, cols * rows + 1):
        image = sample['image']
        figure.add_subplot(rows, cols, idx)
        plt.axis("off")
        plt.imshow(image.squeeze(), cmap="gray")
        sample = next(sample_iter)
    plt.savefig(src_data)
    plt.show()
    
if __name__ == "__main__":
    train_dataset = (get_train_dataset)
    visualize(train_dataset)