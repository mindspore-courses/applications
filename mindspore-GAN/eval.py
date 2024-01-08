import gc
import numpy as np
import matplotlib.pyplot as plt

import mindspore.numpy
from mindspore.common import dtype as mstype
from mindspore.common import set_seed
from mindspore.dataset import MnistDataset
from mindspore import Tensor

from src.GAN_model import *
from src.configs import *


set_seed(1)
reshape = mindspore.ops.Reshape()
log = mindspore.ops.Log()
exp = mindspore.ops.Exp()
cat = mindspore.ops.Concat()
cat2 = mindspore.ops.Concat(1)
squeeze1 = mindspore.ops.Squeeze(1)
sigma = None

# log_mean_exp函数计算平均值
def log_mean_exp(a):
    max_ = a.max(axis=1)
    max2 = reshape(max_, (max_.shape[0], 1))
    return max_ + log(exp(a - max2).mean(1))

# 使用parzen进行模型评估，设计窗口函数
def mind_parzen(x, mu, sigma):
    a = (reshape(x, (x.shape[0], 1, x.shape[-1])) - reshape(mu, (1, mu.shape[0], mu.shape[-1]))) / sigma
    a5 = -0.5 * (a ** 2).sum(2)
    E = log_mean_exp(a5)
    t4 = sigma * np.sqrt(np.pi * 2)
    t5 = np.log(t4)
    Z = mu.shape[1] * t5
    return E - Z

# 通过get_nll函数获得似然值，其中的操作是每次输入一个batch_size,最后对各个batch_size求取平均值
def get_nll(x, samples, sigma, batch_size=10):
    inds = list(range(x.shape[0]))
    n_batches = int(np.ceil(float(len(inds)) / batch_size))

    nlls = Tensor(np.array([]).astype(np.float32))
    for i in range(n_batches):
        nll = mind_parzen(x[inds[i::n_batches]], samples, sigma)
        nlls = cat((nlls, nll))
    return nlls

# 在验证集上进行交叉检验
def cross_validate_sigma(samples, data, sigmas, batch_size):
    lls = Tensor(np.array([]).astype(np.float32))
    for sigma in sigmas:
        print("sigma=", sigma)
        tmp = get_nll(data, samples, sigma, batch_size=batch_size)
        tmp = reshape(tmp.mean(), (1, 1))
        tmp = squeeze1(tmp)
        lls = cat((lls, tmp))
        gc.collect()

    ind = lls.argmax()
    return sigmas[ind]

# 获取有效的验证图像
def get_valid(batch_size_valid=1000):
    dataset = MnistDataset("MNIST_Data/train", "train")
    dataset = dataset.map(
        operations=lambda x: (
            x[-10000:].astype("float32")
        ),
        output_columns=["image"]
    )
    dataset = dataset.batch(batch_size_valid)
    for data in dataset.create_dict_iterator():
        image = data["image"]
        break
    print(type(image))
    image = reshape(image, (image.shape[0], 784))
    return image

# 获取测试图像
def get_test(limit_size):
    dataset_ = MnistDataset("MNIST_Data/test", "test").batch(limit_size)
    for batch_imgs, _ in dataset_.create_tuple_iterator():
        batch_imgs = batch_imgs.astype(mindspore.float32)
        test_data = reshape(batch_imgs, (batch_imgs.shape[0], -1))
        break
    return test_data
    

# 使用对数最大似然进行评估
def parzen(samples):
    ll, se = [1], 1
    shape = samples.shape
    samples = reshape(samples, (shape[0], -1))
    
    valid = get_valid(batch_size_valid=BATCH_SIZE_VALID).asnumpy()
    valid = Tensor(valid / 255)

    sigma_range = np.logspace(-1, 0, num=5)  # 等比数列
    sigma = cross_validate_sigma(samples, valid, sigma_range, batch_size=BATCH_SIZE_TEST)

    # print("Using Sigma: {}".format(sigma))
    gc.collect()

    test_data = get_test(limit_size=1000) / 255
    ll = get_nll(test_data, samples, sigma, batch_size=BATCH_SIZE_TEST)
    se = ll.std() / np.sqrt(test_data.shape[0])

    return ll.mean(), se

test_noise = Tensor(np.random.normal(size=(1000, LATENT_DIM)), dtype=mstype.float32)
llh, epochs = [], []
def eval(n):
    # 导入生成器参数
    model = Generator(latent_dim=LATENT_DIM)
    param_dict = mindspore.load_checkpoint("./result/checkpoints/Generator"+str(n)+".ckpt")
    mindspore.load_param_into_net(model, param_dict)
    # 生成噪音数据
    samples1 = model(test_noise)
    fake_img = Tensor((samples1.asnumpy() * 127.5 + 127.5) / 255)
    fake_img = reshape(fake_img, (fake_img.shape[0], 784))
    # 计算对数最大似然估计
    mean_ll, se_ll = parzen(fake_img)
    epochs.append(n)
    llh.append(mean_ll)
    print("epoch = {}, Log-Likelihood of test set = {}, se: {}".format(n, mean_ll, se_ll))
    
for epoch in range(0, TRAIN_EPOCH, 5):
    eval(epoch)
llh = [i.asnumpy() for i in llh]
plt.plot(epochs, llh)
eval_path = "./result/eval.png"
plt.savefig(eval_path)
plt.show()