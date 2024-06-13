
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


def train(net, ckpk_file_name, device_target='CPU'):

    ms.set_context(device_target=device_target)

    train_dataset = Cifar10Dataset(r"CIFAR10/cifar10/train", usage="train", shuffle=True)  # 训练集
    # test_dataset = ds.Cifar10Dataset("CIFAR10/cifar-10-batches-py", usage="test", shuffle=True)  # 测试集

    train_dataset = datapipe(train_dataset, 32, "train")
    steps = train_dataset.get_dataset_size()
    # test_dataset = datapipe(test_dataset, BATCH_SIZE, "test")
    lr = nn.exponential_decay_lr(0.1,
                                 0.6,
                                 total_step=60 * steps,
                                 step_per_epoch=60,
                                 decay_epoch=1)
    # Adam 优化器
    # opt = nn.Adam(
    #     params=net.get_parameters(),
    #     learning_rate=1e-4
    # )

    opt = nn.Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, 0.9, weight_decay=0.0005)
    # 损失函数
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 初始化模型
    model = Model(net, loss_fn=loss, optimizer=opt, metrics={"Accuracy": nn.Accuracy()})

    # 训练
    model.train(60, train_dataset, callbacks=[LossMonitor(1)])

    ms.save_checkpoint(net, ckpt_file_name=ckpk_file_name)


if __name__ == '__main__':

    ghost_model = GhostNet(width=1.3, in_channels=3, num_classes=10)
    google_model = GoogLeNet(in_channels=3, num_classes=10)

    train(ghost_model, "ghostnet_22.ckpt", device_target='GPU')

    train(google_model,  "googlenet_22.ckpt", device_target='GPU')


