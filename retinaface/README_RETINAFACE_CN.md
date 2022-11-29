# 目录

<!-- TOC -->

- [目录](#目录)
- [Retinaface描述](#Retinaface描述)
- [模型架构](#模型架构)
- [数据集](#数据集)
    - [预训练文件](#anchor架构)

- [特性](#特性)
    - [anchor架构](#anchor架构)
- [环境要求](#环境要求)
- [快速入门](#快速入门)
- [脚本说明](#脚本说明)
    - [脚本及样例代码](#脚本及样例代码)
    - [脚本参数](#脚本参数)
    - [训练过程](#训练过程)
        - [训练](#训练)
    - [评估过程](#评估过程)
        - [评估](#评估)
- [模型描述](#模型描述)
    - [使用流程](#使用流程)
        - [推理](#推理)
- [随机情况说明](#随机情况说明)
- [ModelZoo主页](#modelzoo主页)

<!-- /TOC -->

## Retinaface描述

Retinaface人脸检测模型于2019年提出，应用于WIDER  FACE数据集时效果最佳。RetinaFace论文：RetinaFace: Single-stage Dense Face  Localisation in the  Wild。与S3FD和MTCNN相比，RetinaFace显著提上了小脸召回率，但不适合多尺度人脸检测。为了解决这些问题，RetinaFace采用RetinaFace特征金字塔结构进行不同尺度间的特征融合，并增加了SSH模块。

Retinaface附带一个facealignment2D模块，用于在包含人脸的图片上标记出关键点。该模块可以对含有一张人脸的图片进行独立推理，为了满足有一张图片有多张人脸的情况，这个模块可以读取retinaface识别人脸的输出进行图片的裁剪，即联合推理。该模块的模型所用的权重文件由官方的权重文件直接转换为mindspore的ckpt，所以没有train与eval的阶段。

[论文](https://gitee.com/link?target=https%3A%2F%2Farxiv.org%2Fabs%2F1905.00641v2)：  Jiankang Deng, Jia Guo, Yuxiang Zhou, Jinke Yu, Irene Kotsia, Stefanos Zafeiriou. "RetinaFace: Single-stage Dense Face Localisation in the  Wild". 2019.

## 模型架构

在模型设计上，Retinaface考虑到应用场景的不同，以及为了更综合的考虑网络结构的性能，模型提供了两种不同的骨干网，分别为庞大的ResNet50和方便部署的MobileNet025。在骨干网将特征提取完毕后，模型通过特征金字塔以及SSH模块，分别进行多尺度提取和多任务学习。

![retinaface_structure](./images/retinaface_structure1.png)

最终根据SSH模块的不同输出对图片进行人脸对齐的不同工作，具体工作包含人脸对齐以及人脸关键点检测。其中模型本身可以预测人脸关键点，通过FPN(特征金字塔)、SSH以及anchor结构可以检测不同大小的人脸并进行对齐。

## 数据集

使用的图片数据集：[人脸对齐数据集](http://shuoyang1213.me/WIDERFACE/index.html)

使用的label标注文件：[label](https://drive.google.com/file/d/1vgCABX1JI3NGBzsHxwBXlmRjaLV3NIsG/view)

使用的ground_truth真实值标签：[ground_truth](https://github.com/peteryuX/retinaface-tf2/tree/master/widerface_evaluate/ground_truth)

使用的预训练权重位置：[pretrained_model](https://download.mindspore.cn/vision/retinaface/pretrained_model/)

- 数据集大小：1.85G，共61个类、32203图像、以及393,703个标注人脸
    - 训练集：1.39G，共158,989张标注人脸
    - 测试集：0.34G，共39,496张标注人脸
- 数据格式：RGB
- 图片数据集、标签、ground truth以及预训练权重下载完毕后需要在项目目录下组织成以下形式

 ```bash
├── data/
    ├── widerface/
        ├── ground_truth/
        │   ├──wider_easy_val.mat
        │   ├──wider_face_val.mat
        │   ├──wider_hard_val.mat
        │   ├──wider_medium_val.mat
        ├── train/
        │   ├──images/
        │   │   ├──0--Parade/
        │   │   │   ├──0_Parade_marchingband_1_5.jpg
        │   │   │   ├──...
        │   │   ├──.../
        │   ├──label.txt
        ├── val/
        │   ├──images/
        │   │   ├──0--Parade/
        │   │   │   ├──0_Parade_marchingband_1_20.jpg
        │   │   │   ├──...
        │   │   ├──.../
        │   ├──label.txt
├── pretrained_model
    ├──resnet_pretrain.ckpt        // resnet骨干网预训练权重
    ├──mobilenet_pretrain.ckpt     // mobile骨干网预训练权重
    ├──FaceAlignment2D.ckpt        // facealignment2D网络预训练权重(4.8MiB)
 ```

## 特性

### anchor架构

目标检测一般都是采用anchor_based的方法，大致可以分为单阶段检测器和双阶段检测器。它们都是在一张图片上放置大量的预先定义好的 anchor boxes，然后预测其类别，优化这些anchor boxes的坐标，最终将这些优化后的 anchor boxes作为检测结果输出。本案例就是在单阶段检测器中应用anchor。

keypoint-based methods：这类 anchor-free 方法首先定位到预先定义或自学习的关键点，然后生成边框来检测物体。CornerNet 通过一对关键点（左上角和右下角）来检测物体的边框，CornerNet-Lite 引入了 CornerNet-Saccade 和 CornerNet-Squeeze 来提升其速度。Grid R-CNN 的第二个阶段利用FCN的位置敏感的优点来预测网格点，然后再判断边框、定位物体。ExtremeNet 检测物体的4个点（最上面、最左面、最下面、最右面）以及一个中心点来生成物体的边框。Zhu 等人利用关键点估计来找到物体的中心点，然后回归出其他的属性，包括大小、三维位置、朝向、姿态等。CenterNet 扩展了CornerNet，通过三个点而不是两个点来提升精度和召回率。RepPoints 将物体表示为一个样本点的集合，通过约束物体的空间范围、强调语义重要的局部区域来学习。

本文使用了keypoint-based的方法，对于输入的640X640的图像缩放到80X80，40X40，20X20三个尺度，同时在缩放后的每个特征图上生成2个不同大小的正方形anchor用于目标检测。

## 环境要求

- 硬件（Ascend/GPU/CPU）
    - Retinaface主网络使用Ascend/GPU/CPU处理器来搭建硬件环境。
    - 在使用FaceAlignment2D模块的时候只可在Ascend/GPU环境下工作，该模型中的PRelu算子不支持CPU环境。
- 框架
    - [MindSpore](https://www.mindspore.cn/install/en)
- 如需查看详情，请参见如下资源：
    - [MindSpore教程](https://www.mindspore.cn/tutorials/zh-CN/master/index.html)
    - [MindSpore Python API](https://www.mindspore.cn/docs/api/zh-CN/master/index.html)

## 快速入门

通过官方网站安装MindSpore后，您可以按照如下步骤进行训练和评估：

- Ascend处理器环境运行

  ```shell
  # 运行训练示例 配置项通过argparse内置在文件中
  python retinface_train.py

  # 运行评估示例
  python retinaface_eval.py

  # 运行推理示例
  python retinaface_infer.py

  # 人脸对齐模块推理示例
  python facealignment_infer.py --pre_trained ./pretrained_model/FaceAlignment2D.ckpt --clipped_path ../images/facealignment/source/ --output_path ../images/facealignment/output --device_target Ascend
  ```

### 脚本及样例代码

```text
├── model_zoo
    ├── retinaface
        ├── images                              // 存储案例以及md所需的图片
        ├── src
        |    ├── data
        |        ├── widerface
        |        |   ├── train
        |        |       ├──images
        |        |       │   ├──0--Parade
        |        |       │   │   ├──0_Parade_marchingband_1_5.jpg
        |        |       │   │   ├──...
        |        |       │   ├──...
        |        |       ├──label.txt
        |        |   ├── val
        |        |       ├──images
        |        |       │   ├──0--Parade
        |        |       │   │   ├──0_Parade_marchingband_1_20.jpg
        |        |       │   │   ├──...
        |        |       │   ├──...
        |        |       ├──label.txt
        |        ├── ground_truth
        |        │   ├──wider_easy_val.mat
        |        │   ├──wider_face_val.mat
        |        │   ├──wider_hard_val.mat
        |        │   ├──wider_medium_val.mat
        │    ├── pretrained_model
        │        ├──resnet_pretrain.ckpt        // resnet骨干网预训练权重
        │        ├──mobilenet_pretrain.ckpt     // mobile骨干网预训练权重
        │        ├──FaceAlignment2D.ckpt        // facealignment2D网络预训练权重(4.8MiB)
        │    ├── model
        │        ├──facealignment.py            // 人脸对齐模块网络
        │        ├──head.py                     // RetinaFace BoxHead，用于预测人脸框高、宽以及中心位置
        │        ├──loss_cell.py                // 模型loss定义
        │        ├──mobilenet025.py             // mobilenet骨干网
        │        ├──resnet50.py                 // resnet骨干网
        │        ├──retinaface.py               // 模型基础模块，包含ssh等
        │    ├── process_datasets
        │        ├──widerface.py                // 数据集创建
        │        ├──pre_process.py              // 数据集预处理
        │    ├── utils
        │        ├──config.py                   // 人脸对齐模块配置文件
        │        ├──initialize.py               // kaiming均匀初始化模块
        │        ├──lr_schedule.py              // 学习率调整模块
        │        ├──detection.py                // 人脸检测方法
        │        ├──detection_engine.py         // 人脸检测模块
        │        ├──draw_prediction.py          // 将检测结果绘画到图片上
        │        ├──multiboxloss.py             // multiboxloss定义
        │        ├──timer.py                    // 计时模块
        │    ├── facealignment_infer.py         // 人脸对齐推理脚本
        │    ├── retinaface_eval.py             // 评估脚本
        │    ├── retinaface_infer.py            // 微调训练脚本
        │    ├── retinaface_train.py            // 预训练脚本
        ├── retinaface.ipynb                    // retinaface模型jupyter案例
        ├── README_RETINAFACE_CN.md             // retinaface模型相关说明
```

### 脚本参数

retinaface和widerface数据集配置。

```python
    # 训练配置项，位于retinaface_train.py文件中
    parser = argparse.ArgumentParser(description='train')
    # 模型骨干网
    parser.add_argument('--backbone', default='mobilenet025', type=str, choices=['resnet50', 'mobilenet025'])
    parser.add_argument('--variance', default=[0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--clip', default=False, type=bool)
    parser.add_argument('--loc_weight', default=2.0, type=float)
    parser.add_argument('--class_weight', default=1.0, type=float)
    parser.add_argument('--landm_weight', default=1.0, type=float)
    #  模型训练的batch size
    parser.add_argument('--batch_size', default=8, type=int)
    #  生成的anchor数量
    parser.add_argument('--num_anchor', default=16800, type=int)
    #  图片大小
    parser.add_argument('--image_size', default=640, type=int)
    parser.add_argument('--match_thresh', default=0.35, type=float)
    #  输入输出的通道数
    parser.add_argument('--in_channel', default=32, type=int)
    parser.add_argument('--out_channel', default=64, type=int)
    parser.add_argument('--seed', default=1, type=int)
    #  epoch
    parser.add_argument('--epoch', default=120, type=int)
    #  lr
    parser.add_argument('--initial_lr', default=0.005, type=float)
    #  训练过程中模型权重的保存位置
    parser.add_argument('--ckpt_path', default='./mobilenet_ckpts/', type=str)
    parser.add_argument('--save_checkpoint_steps', default=100, type=int)
    #  保存模型权重的数量
    parser.add_argument('--keep_checkpoint_max', default=3, type=int)
    #  训练集位置
    parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', type=str)
    #  如果有预训练模型需要进行配置
    parser.add_argument('--pretrain', default=False, type=bool)
    parser.add_argument('--pretrain_path', default='./pretrained_model/resnet_pretrain.ckpt', type=str)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--negative_ratio', default=7, type=int)
    parser.add_argument('--device_target', default='Ascend', type=str)
    parser.add_argument('--device_id', default=0, type=str)
    parser.add_argument("--decay1", type=int, default=70)
    parser.add_argument("--decay2", type=int, default=90)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lr_type", type=str, default="dynamic_lr", choices=['dynamic_lr', 'cosine_annealing'])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_epoch", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--T_max", type=int, default=50)
    parser.add_argument("--eta_min", type=float, default=0.0)
    parser.add_argument("--loss_scale", type=int, default=1)
    parser.add_argument("--optim", type=str, default="sgd", choices=['sgd', 'momentum'])
    parser.add_argument("--num_workers", type=int, default=2)
    #  验证配置项，位于retinaface_valid.py文件中
    #  用于验证模型的保存位置
    parser.add_argument('--val_model', default='./mobilenet_ckpts/RetinaFace-120_1609.ckpt', type=str)
    parser.add_argument('--val_origin_size', default=False, type=bool)
    #  用于验证的数据集的保存位置
    parser.add_argument('--val_dataset_folder', default='./data/widerface/valid/', type=str)
    parser.add_argument('--val_save_result', default=True, type=bool)
    #  nms、iou的阈值设定
    parser.add_argument('--val_confidence_threshold', default=0.02, type=float)
    parser.add_argument('--val_nms_threshold', default=0.4, type=float)
    parser.add_argument('--val_iou_threshold', default=0.5, type=float)
    # 验证时计算结果的保存位置
    parser.add_argument('--val_predict_save_folder', default='./data/widerface_result', type=str)
    #  验证集ground_truth文件的路径
    parser.add_argument('--val_gt_dir', default='./data/ground_truth/', type=str)
    parser.add_argument('--min_sizes', default='[[16, 32], [64, 128], [256, 512]]', type=str)
    parser.add_argument('--steps', default=[8, 16, 32], type=float, nargs='+')
    parser.add_argument('--image_average', default=(104.0, 117.0, 123.0), type=float, nargs='+')
    parser.add_argument('--target_size', default=1600, type=int)
    parser.add_argument('--max_size', default=2176, type=int)
    #  推理配置项，位于retinaface_infer.py文件中
    #  需要推理的图片位置
    parser.add_argument('--img_folder', default='./input_image', type=str)
    #  推理完毕图片的保存位置
    parser.add_argument('--draw_save_folder', default='./infer_image', type=str)
    parser.add_argument('--conf_thre', default=0.4, type=float)
```

如果使用resnet骨干网进行实验则需要将配置中的backbone更改为resnet025，将num_anchor数值更改为29126，将image_size的数值更改为840，将in_channel更改为256，将out_channel更改为256，具体的部分训练配置如下所示，其他部分相同的设置项也需要对应更改：

```python
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--backbone', default='resnet50', type=str, choices=['resnet50', 'mobilenet025'])
    parser.add_argument('--variance', default=[0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--clip', default=False, type=bool)
    parser.add_argument('--loc_weight', default=2.0, type=float)
    parser.add_argument('--class_weight', default=1.0, type=float)
    parser.add_argument('--landm_weight', default=1.0, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_anchor', default=29126, type=int)
    parser.add_argument('--image_size', default=840, type=int)
    parser.add_argument('--match_thresh', default=0.35, type=float)
    parser.add_argument('--in_channel', default=256, type=int)
    parser.add_argument('--out_channel', default=256, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--epoch', default=120, type=int)
    parser.add_argument('--initial_lr', default=0.005, type=float)
    parser.add_argument('--ckpt_path', default='./resnet_ckpts/', type=str)
    parser.add_argument('--save_checkpoint_steps', default=100, type=int)
    parser.add_argument('--keep_checkpoint_max', default=3, type=int)
    parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', type=str)
    parser.add_argument('--pretrain', default=False, type=bool)
    parser.add_argument('--pretrain_path', default='./pretrained_model/resnet_pretrain.ckpt', type=str)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--negative_ratio', default=7, type=int)
    parser.add_argument('--device_target', default='Ascend', type=str)
    parser.add_argument('--device_id', default=0, type=str)
    parser.add_argument("--decay1", type=int, default=20)
    parser.add_argument("--decay2", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lr_type", type=str, default="dynamic_lr", choices=['dynamic_lr', 'cosine_annealing'])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_epoch", type=int, default=-1)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--T_max", type=int, default=50)
    parser.add_argument("--eta_min", type=float, default=0.0)
    parser.add_argument("--loss_scale", type=int, default=1)
    parser.add_argument("--optim", type=str, default="sgd", choices=['sgd', 'momentum'])
```

FaceAlignment2D部分参数配置

```python
    parser.add_argument('--mode', type=str, default='standalone', help='Infer Work Alone / work with Retinaface')
    parser.add_argument('--pre_trained', type=str, default=None, help='Pretrained checkpoint path')
    parser.add_argument('--device_target', type=str, default="GPU", help='run device_target, GPU or Ascend')
    parser.add_argument('--raw_image_path', type=str, default=None, help='Raw Img Folder Path')
    parser.add_argument('--json_path', type=str, default=None, help='json file generated bu retinaface')
    parser.add_argument('--clipped_path', type=str, default=None, help='Clipped Picture Output Path')
    parser.add_argument('--output_path', type=str, default=None, help='Predict Result Output Path')
    parser.add_argument('--device_id', type=int, default=0, help='Device id')
```

### 训练过程

#### 训练

- Ascend处理器环境运行

  ```shell
  python retinaface_train.py
  ```

  模型检查点保存在运行参数cfg['ckpt_path']目录下。

### 评估过程

#### 评估

- 在Ascend环境运行时评估widerface数据集

  在运行以下命令之前，请检查用于评估的检查点路径。请将检查点路径val_modal设置为绝对全路径，例如“/root/rtf/new_code/data/RetinaFace-6_1609.ckpt”。

  ```bash
  # 运行评估示例
  python retinaface_eval.py
  ```

  上述python命令将在前台运行，您可以通过cmd窗口文件结果，也可以通过“python retinaface_eval.py > eval.log 2>&1 &”将日志保存在eval.log文件中。测试数据集的准确性如下：

  ```bash
  Easy   Val AP : 0.7606
  Medium Val AP : 0.7307
  Hard   Val AP : 0.4596
  ```

## 模型描述

### 使用流程

#### 推理

如果您需要使用此训练模型在GPU、Ascend 910、Ascend 310等多个硬件平台上进行推理，可参考此[链接](https://www.mindspore.cn/tutorials/experts/zh-CN/master/infer/inference.html)。下面是操作步骤示例：

- Ascend处理器环境运行

  ```python
  # 通过函数获取模型训练需要的参数，以下代码为retinaface_infer.py中主要代码的截取
  cfg = parse_args()

  # 设置上下文
  context.set_context(mode=context.GRAPH_HOME, device_target=args.device_target)
  context.set_context(device_id=args.device_id)

  # 获取需要推理的图片
  test_dataset = read_input_images(cfg['img_folder'])
  num_images = len(test_dataset)

  # 定义模型,以resnet50骨干网为例
  backbone = resnet50()
  network = RetinaFace(phase='predict', backbone=backbone)
  backbone.set_train(False)
  network.set_train(False)

  # 加载预训练模型
  param_dict = ms.load_checkpoint(cfg['val_model'])
  network.init_parameters_data()
  ms.load_param_into_net(network, param_dict)
  net.set_train(False)

  # 实例化先验框检测类
  detection = DetectionEngine(cfg)
  ...

  # 执行推理并保存推理结果
  detection.detect(boxes, ldm, confs, resize, scale, ldm_scale, img_name, priors)
  draw_image(detection.results['infer'], cfg['img_folder'], cfg['draw_save_folder'], cfg['conf_thre'])
  ```

## 随机情况说明

每个任务的启动文件中都在配置项中规定了固定的随机数种子，如retinaface_train.py中，parse_args()函数中的seed配置项。

## ModelZoo主页

 请浏览官网[主页](https://gitee.com/mindspore/models)。

## 详细操作流程

本流程大致分为两个部分，第一个部分是从华为git仓库克隆代码，第二部分是notebook训练、验证和测试

### 1.代码下载

通过命令行使用以下命令将代码下载在选定文件夹中

运行完毕后该文件夹下只会存在一个retinaface_course文件夹

```bash
git clone http://8.130.182.184:9080/liuh/retinaface_course.git
```

在./retinaface_course/application_example中新建一个data文件夹，用于存放数据集

登录obs，在obs://datasets/widerface/目录下将ground_truth和widerface文件夹下载到data文件中；在obs://application/ckpts/retinaface/train_00/目录下找到RetinaFace-120_1609.ckpt文件，并将其下载到data文件夹中。

最后将data和retinaface放在文件夹code1中，并将code1文件夹打成zip压缩包。code1文件目录如下

```bash
├── model_zoo
    ├── retinaface
        ├── images                              // 存储案例以及md所需的图片
        ├── src
        |    ├── data
        |        ├── widerface
        |        |   ├── train
        |        |       ├──images
        |        |       │   ├──0--Parade
        |        |       │   │   ├──0_Parade_marchingband_1_5.jpg
        |        |       │   │   ├──...
        |        |       │   ├──...
        |        |       ├──label.txt
        |        |   ├── val
        |        |       ├──images
        |        |       │   ├──0--Parade
        |        |       │   │   ├──0_Parade_marchingband_1_20.jpg
        |        |       │   │   ├──...
        |        |       │   ├──...
        |        |       ├──label.txt
        |        ├── ground_truth
        |        │   ├──wider_easy_val.mat
        |        │   ├──wider_face_val.mat
        |        │   ├──wider_hard_val.mat
        |        │   ├──wider_medium_val.mat
        │    ├── pretrained_model
        │        ├──resnet_pretrain.ckpt        // resnet骨干网预训练权重
        │        ├──mobilenet_pretrain.ckpt     // mobile骨干网预训练权重
        │        ├──FaceAlignment2D.ckpt        // facealignment2D网络预训练权重(4.8MiB)
        │    ├── model
        │        ├──facealignment.py            // 人脸对齐模块网络
        │        ├──head.py                     // RetinaFace BoxHead，用于预测人脸框高、宽以及中心位置
        │        ├──loss_cell.py                // 模型loss定义
        │        ├──mobilenet025.py             // mobilenet骨干网
        │        ├──resnet50.py                 // resnet骨干网
        │        ├──retinaface.py               // 模型基础模块，包含ssh等
        │    ├── process_datasets
        │        ├──widerface.py                // 数据集创建
        │        ├──pre_process.py              // 数据集预处理
        │    ├── utils
        │        ├──config.py                   // 人脸对齐模块配置文件
        │        ├──initialize.py               // kaiming均匀初始化模块
        │        ├──lr_schedule.py              // 学习率调整模块
        │        ├──detection.py                // 人脸检测方法
        │        ├──detection_engine.py         // 人脸检测模块
        │        ├──draw_prediction.py          // 将检测结果绘画到图片上
        │        ├──multiboxloss.py             // multiboxloss定义
        │        ├──timer.py                    // 计时模块
        │    ├── facealignment_infer.py         // 人脸对齐推理脚本
        │    ├── retinaface_eval.py             // 评估脚本
        │    ├── retinaface_infer.py            // 微调训练脚本
        │    ├── retinaface_train.py            // 预训练脚本
        ├── retinaface.ipynb                    // retinaface模型jupyter案例
        ├── README_RETINAFACE_CN.md             // retinaface模型相关说明
```

在上传之前需要在工作目录下创建notebook，镜像需要选用retinaface_ms17_cv2:0.0.2镜像。

将zip压缩包通过jupyter自带的文件上传上传到notebook的工作目录下，选择使用obs中转进行上传，上传完毕后通过如下命令将zip文件在notebook中解压：

```bash
unzip code1.zip -d ./code1
```

### 2.模型训练

进入./code/retinaface/src目录下，打开retinaface_train.py文件，并用以下代码替换改文件的parse_args函数:

```python
def parse_args():
    """Parse configuration arguments for training."""
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--backbone', default='mobilenet025', type=str, choices=['resnet50', 'mobilenet025'])
    parser.add_argument('--variance', default=[0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--clip', default=False, type=bool)
    parser.add_argument('--loc_weight', default=2.0, type=float)
    parser.add_argument('--class_weight', default=1.0, type=float)
    parser.add_argument('--landm_weight', default=1.0, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_anchor', default=16800, type=int)
    parser.add_argument('--image_size', default=640, type=int)
    parser.add_argument('--match_thresh', default=0.35, type=float)
    parser.add_argument('--in_channel', default=32, type=int)
    parser.add_argument('--out_channel', default=64, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--epoch', default=120, type=int)
    parser.add_argument('--initial_lr', default=0.005, type=float)
    parser.add_argument('--ckpt_path', default='./mobilenet_ckpts/', type=str)
    parser.add_argument('--save_checkpoint_steps', default=100, type=int)
    parser.add_argument('--keep_checkpoint_max', default=3, type=int)
    parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', type=str)
    parser.add_argument('--pretrain', default=False, type=bool)
    parser.add_argument('--pretrain_path', default='./pretrained_model/mobilenet_pretrain.ckpt', type=str)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--negative_ratio', default=7, type=int)
    parser.add_argument('--device_target', default='Ascend', type=str)
    parser.add_argument('--device_id', default=0, type=str)
    parser.add_argument("--decay1", type=int, default=70)
    parser.add_argument("--decay2", type=int, default=90)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lr_type", type=str, default="dynamic_lr", choices=['dynamic_lr', 'cosine_annealing'])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_epoch", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--T_max", type=int, default=50)
    parser.add_argument("--eta_min", type=float, default=0.0)
    parser.add_argument("--loss_scale", type=int, default=1)
    parser.add_argument("--optim", type=str, default="sgd", choices=['sgd', 'momentum'])
    parser.add_argument("--num_workers", type=int, default=2)
    return vars(parser.parse_args(()))
```

resnet50骨干网需要换成下面的参数

```python
def parse_args():
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--backbone', default='resnet50', type=str, choices=['resnet50', 'mobilenet025'])
    parser.add_argument('--variance', default=[0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--clip', default=False, type=bool)
    parser.add_argument('--loc_weight', default=2.0, type=float)
    parser.add_argument('--class_weight', default=1.0, type=float)
    parser.add_argument('--landm_weight', default=1.0, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--num_anchor', default=29126, type=int)
    parser.add_argument('--image_size', default=840, type=int)
    parser.add_argument('--match_thresh', default=0.35, type=float)
    parser.add_argument('--in_channel', default=256, type=int)
    parser.add_argument('--out_channel', default=256, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--epoch', default=120, type=int)
    parser.add_argument('--initial_lr', default=0.005, type=float)
    parser.add_argument('--ckpt_path', default='./resnet_ckpts/', type=str)
    parser.add_argument('--save_checkpoint_steps', default=100, type=int)
    parser.add_argument('--keep_checkpoint_max', default=3, type=int)
    parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', type=str)
    parser.add_argument('--pretrain', default=False, type=bool)
    parser.add_argument('--pretrain_path', default='./pretrained_model/resnet_pretrain.ckpt', type=str)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--negative_ratio', default=7, type=int)
    parser.add_argument('--device_target', default='Ascend', type=str)
    parser.add_argument('--device_id', default=0, type=str)
    parser.add_argument("--decay1", type=int, default=20)
    parser.add_argument("--decay2", type=int, default=40)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--lr_type", type=str, default="dynamic_lr", choices=['dynamic_lr', 'cosine_annealing'])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup_epoch", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--T_max", type=int, default=50)
    parser.add_argument("--eta_min", type=float, default=0.0)
    parser.add_argument("--loss_scale", type=int, default=1)
    parser.add_argument("--optim", type=str, default="sgd", choices=['sgd', 'momentum'])
    parser.add_argument("--num_workers", type=int, default=2)
    return vars(parser.parse_args(()))
```

训练参数具体含义可以参考上一部分，[脚本参数](#脚本参数)

在控制台输入以下代码进行训练：

```bash
python retinaface_train.py
```

### 3.模型验证

进入./code/retinaface/src目录下，打开retinaface_eval.py文件，用以下代码替换parse_args函数:

```python
def parse_args():
    """Parse configuration arguments for evaluating."""
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--backbone', default='mobilenet025', type=str, choices=['resnet50', 'mobilenet025'])
    parser.add_argument('--val_model', default='./mobilenet_ckpts/RetinaFace-120_1609.ckpt', type=str)
    parser.add_argument('--val_origin_size', default=False, type=bool)
    parser.add_argument('--val_dataset_folder', default='./data/widerface/valid/', type=str)
    parser.add_argument('--val_save_result', default=True, type=bool)
    parser.add_argument('--val_confidence_threshold', default=0.02, type=float)
    parser.add_argument('--val_nms_threshold', default=0.4, type=float)
    parser.add_argument('--val_iou_threshold', default=0.5, type=float)
    parser.add_argument('--variance', default=[0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--val_predict_save_folder', default='./data/widerface_result', type=str)
    parser.add_argument('--val_gt_dir', default='./data/ground_truth/', type=str)
    parser.add_argument('--min_sizes', default='[[16, 32], [64, 128], [256, 512]]', type=str)
    parser.add_argument('--steps', default=[8, 16, 32], type=float, nargs='+')
    parser.add_argument('--clip', default=False, type=bool)
    parser.add_argument('--image_average', default=(104.0, 117.0, 123.0), type=float, nargs='+')
    parser.add_argument('--target_size', default=1600, type=int)
    parser.add_argument('--max_size', default=2176, type=int)
    parser.add_argument('--in_channel', default=32, type=int)
    parser.add_argument('--out_channel', default=64, type=int)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--device_target', default='Ascend', type=str)
    return vars(parser.parse_args())
```

```python
def parse_args():
    """Parse configuration arguments for evaluating."""
    parser = argparse.ArgumentParser(description='eval')
    parser.add_argument('--backbone', default='resnet50', type=str, choices=['resnet50', 'mobilenet025'])
    parser.add_argument('--val_model', default='./resnet_ckpts/RetinaFace-120_1609.ckpt', type=str)
    parser.add_argument('--val_origin_size', default=False, type=bool)
    parser.add_argument('--val_dataset_folder', default='./data/widerface/valid/', type=str)
    parser.add_argument('--val_save_result', default=True, type=bool)
    parser.add_argument('--val_confidence_threshold', default=0.02, type=float)
    parser.add_argument('--val_nms_threshold', default=0.4, type=float)
    parser.add_argument('--val_iou_threshold', default=0.5, type=float)
    parser.add_argument('--variance', default=[0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--val_predict_save_folder', default='./data/widerface_result', type=str)
    parser.add_argument('--val_gt_dir', default='./data/ground_truth/', type=str)
    parser.add_argument('--min_sizes', default='[[16, 32], [64, 128], [256, 512]]', type=str)
    parser.add_argument('--steps', default=[8, 16, 32], type=float, nargs='+')
    parser.add_argument('--clip', default=False, type=bool)
    parser.add_argument('--image_average', default=(104.0, 117.0, 123.0), type=float, nargs='+')
    parser.add_argument('--target_size', default=1600, type=int)
    parser.add_argument('--max_size', default=2176, type=int)
    parser.add_argument('--in_channel', default=256, type=int)
    parser.add_argument('--out_channel', default=256, type=int)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--device_target', default='Ascend', type=str)
    return vars(parser.parse_args())
```

在控制台输入以下代码进行验证：

```bash
python retinaface_eval.py
```

### 4.模型推理

进入./code/retinaface目录下，创建input_image和infer_image两个文件夹，其中input_image用于存放需要推理的照片，可随意上传。

进入./code/retinaface/src目录下,用以下代码替换parse_args函数:

```python
def parse_args():
    """Parse configuration arguments for inference."""
    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--backbone', default='mobilenet025', type=str, choices=['resnet50', 'mobilenet025'])
    parser.add_argument('--variance', default=[0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--val_model', default='./mobilenet_ckpts/RetinaFace-120_1609.ckpt', type=str)
    parser.add_argument('--val_origin_size', default=False, type=bool)
    parser.add_argument('--val_confidence_threshold', default=0.02, type=float)
    parser.add_argument('--val_nms_threshold', default=0.4, type=float)
    parser.add_argument('--val_iou_threshold', default=0.5, type=float)
    parser.add_argument('--val_predict_save_folder', default='./data/widerface_result', type=str)
    parser.add_argument('--val_gt_dir', default='./data/ground_truth/', type=str)
    parser.add_argument('--img_folder', default='./input_image', type=str)
    parser.add_argument('--draw_save_folder', default='./infer_image', type=str)
    parser.add_argument('--conf_thre', default=0.4, type=float)
    parser.add_argument('--min_sizes', default='[[16, 32], [64, 128], [256, 512]]', type=str)
    parser.add_argument('--steps', default=[8, 16, 32], type=float, nargs='+')
    parser.add_argument('--clip', default=False, type=bool)
    parser.add_argument('--image_average', default=(104.0, 117.0, 123.0), type=float, nargs='+')
    parser.add_argument('--target_size', default=1600, type=int)
    parser.add_argument('--max_size', default=2176, type=int)
    parser.add_argument('--in_channel', default=32, type=int)
    parser.add_argument('--out_channel', default=64, type=int)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--device_target', default='Ascend', type=str)
    return vars(parser.parse_args())
```

resnet50骨干网需要换成下面的参数

```python
def parse_args():
    """Parse configuration arguments for inference."""
    parser = argparse.ArgumentParser(description='infer')
    parser.add_argument('--backbone', default='resnet50', type=str, choices=['resnet50', 'mobilenet025'])
    parser.add_argument('--variance', default=[0.1, 0.2], type=float, nargs='+')
    parser.add_argument('--val_model', default='./resnet_ckpts/RetinaFace-120_1609.ckpt', type=str)
    parser.add_argument('--val_origin_size', default=False, type=bool)
    parser.add_argument('--val_confidence_threshold', default=0.02, type=float)
    parser.add_argument('--val_nms_threshold', default=0.4, type=float)
    parser.add_argument('--val_iou_threshold', default=0.5, type=float)
    parser.add_argument('--val_predict_save_folder', default='./data/widerface_result', type=str)
    parser.add_argument('--val_gt_dir', default='./data/ground_truth/', type=str)
    parser.add_argument('--img_folder', default='./input_image', type=str)
    parser.add_argument('--draw_save_folder', default='./infer_image', type=str)
    parser.add_argument('--conf_thre', default=0.4, type=float)
    parser.add_argument('--min_sizes', default='[[16, 32], [64, 128], [256, 512]]', type=str)
    parser.add_argument('--steps', default=[8, 16, 32], type=float, nargs='+')
    parser.add_argument('--clip', default=False, type=bool)
    parser.add_argument('--image_average', default=(104.0, 117.0, 123.0), type=float, nargs='+')
    parser.add_argument('--target_size', default=1600, type=int)
    parser.add_argument('--max_size', default=2176, type=int)
    parser.add_argument('--in_channel', default=256, type=int)
    parser.add_argument('--out_channel', default=256, type=int)
    parser.add_argument('--device_id', default=0, type=int)
    parser.add_argument('--device_target', default='Ascend', type=str)
    return vars(parser.parse_args())
```

在控制台输入以下代码进行推理：

```bash
python retinaface_infer.py
```

对于FaceAlignment2D模块的推理示例，如下：

```bash
cd ./application_example/retinaface/src
python facealignment_infer.py --pre_trained ./pretrained_model/FaceAlignment2D.ckpt --clipped_path ../images/facealignment/source/ --output_path ../images/facealignment/output --device_target Ascend
```

如果需要使用Retinaface输出的识别结果，如下操作：

```bash
python facealignment_infer.py --mode retinaface --pre_trained ./pretrained_model/FaceAlignment2D.ckpt --raw_image_path 原始图片文件目录 --json_path Retinaface识别图片后生成的json --clipped_path 中间文件夹用于存放裁剪的图片 --output_path 输出识别结果的目录
```