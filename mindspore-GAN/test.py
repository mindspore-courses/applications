import mindspore
from mindspore import Tensor
from mindspore.common import dtype as mstype
from data_loader import *
from src.configs import *
from src.GAN_model import *

# 从文件中获取模型参数并加载到网络中
model = Generator(latent_dim=100)
param_dict = mindspore.load_checkpoint("./result/checkpoints/Generator170.ckpt")
mindspore.load_param_into_net(model, param_dict)

model.set_train(True)

# 生成图片
test_noise = Tensor(np.random.normal(0, 1, (25, 100)).astype(np.float32))
imgs = model(test_noise).transpose(0, 2, 3, 1).asnumpy()

fig = plt.figure(figsize=(5, 5), dpi=120)
for i in range(25):
    fig.add_subplot(5, 5, i+1)
    plt.axis("off")
    plt.imshow(imgs[i].squeeze(), cmap="gray")
plt.show()
plt.savefig("result/ckpt170_gen.png")