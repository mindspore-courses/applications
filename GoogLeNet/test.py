
import mindspore as ms
import mindspore.nn as nn

from GhostNet import GhostNet
from GoogLeNet import GoogLeNet

import mindspore.dataset as ds
from mindspore.dataset import vision, transforms
from mindspore import Model, LossMonitor, TimeMonitor
from mindspore.dataset import Cifar10Dataset


def datapipe(dataset, batch_size,usage):
    image_transforms=[]
    if usage=="train":
        image_transforms = [
            vision.RandomCrop((32, 32), (4, 4, 4, 4)), # 随机裁剪
            vision.RandomHorizontalFlip(), # 随机水平翻转
            vision.Resize((224, 224)),
            vision.Rescale(1.0 / 255.0,0),
            vision.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            vision.HWC2CHW()
        ]
    elif usage=="test":
        image_transforms = [
            vision.Resize((224, 224)),
            vision.Rescale(1.0 / 255.0,0),
            vision.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)),
            vision.HWC2CHW()
        ]
    label_transform = transforms.TypeCast(ms.int32)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(2) #增强操作重复两次
    return dataset


def test(net, ckpk_file_name, device_target='CPU'):

    ms.set_context(device_target=device_target)

    test_dataset = Cifar10Dataset(r"CIFAR10/cifar10/test", usage="test", shuffle=False)  # 测试集

    test_dataset = datapipe(test_dataset, 32, "test")

    # 损失函数
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 加载已经保存的用于测试的模型
    param_dict = ms.load_checkpoint(ckpk_file_name)
    # 加载参数到网络中
    ms.load_param_into_net(net, param_dict)

    model = ms.Model(net, loss, metrics={"acc"})


    # 测试
    acc = model.eval(test_dataset)
    print("参数路径：{}\n{}".format(ckpk_file_name, acc))


if __name__ == '__main__':

    ghost_model = GhostNet(width=1.3, in_channels=3, num_classes=10)
    google_model = GoogLeNet(in_channels=3, num_classes=10)

    test(ghost_model, "ghostnet.ckpt", device_target='GPU')

    test(google_model,  "googlenet.ckpt", device_target='GPU')


