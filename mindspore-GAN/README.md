# GAN based on MindSpore

> [paperswithcode相关链接](https://paperswithcode.com/paper/generative-adversarial-networks)  

使用Mindspore2.0的面向对象和函数式编程特性实现

有关模型的详细介绍请参见可直接运行的 [GAN.ipynb](GAN.ipynb) 或直接运行 [train.py](https://github.com/xxayt/mindspore-GAN/blob/main/train.py) 亦可

在  [`result/checkpoints`](result/checkpoints) 文件夹下保存部分了训练完成的生成器和判别器模型的参数

此项目文件目录介绍如下：

```text
./mindspore_GAN
├─download_data.py            # 数据下载脚本
├─eval.py                     # 评估模型参数效果
├─data_loader.py              # 加载数据集
├─GAN.ipynb                   # 可独立运行总项目的notebook
├─structure.png               # GAN模型结构图
├─test.py                     # 测试模型参数
├─train.py                    # 训练
│      
├─data                        # 数据集
│
├─result
│  ├─gan_mnist.gif           # 保存训练中动图
│  ├─iter.png                # Loss迭代关系图
│  ├─src_data.png            # 抽取显示原始数据集图
│  ├─ckpt170_gen.png         # 训练170轮产生的生成器生成结果图片
│  ├─images\                 # 生成器生成结果图片
│  │
│  └─checkpoints\            # 模型参数
│
└─src
    ├─configs.py              # 超参数设置
    ├─GAN_model.py            # 模型网络结构
    ├─loss.py                 # 损失函数设置
    └─utils.py                # 其他函数代码
```
