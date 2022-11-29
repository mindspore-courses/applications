# MobileViT

Self-attention models, especially vision transformers, are an alternative to Convolutional Neural Networks (CNN) for learning visual representations. Briefly, ViT divides an image into a series of non-overlapping patches, and then uses multi-head Self-attention in transformers to learn inter-patch representations. ViT has a wide range of applications in the field of computer vision, but large network parameters and delays affect its application in real-world scenarios (resource-constrained). Therefore, more and more task scenarios require the use of lightweight ViT. MobileViT is a lightweight, general purpose transformer vision. MobileViT proposes a different perspective, using transformers as convolutions to process information. The results show that MobileViT significantly outperforms CNN and ViT based networks on different tasks and datasets.

## Pretrained model

Model trained by MindSpore:

| Model         | Parameters | Top-1  | Top-5  | ckpt                                                                           |
|---------------|------------|--------|--------|--------------------------------------------------------------------------------|
| MobileViT-XXS | 1.3 M      | 62.184 | 84.292 | [ckpt](https://download.mindspore.cn/vision/cyclegan/apple/mobilevit_xxs.ckpt) |

Model trained by PyTorch:

| Model         | Parameters | Top-1 | pt                                                                                                        |
|---------------|------------|-------|-----------------------------------------------------------------------------------------------------------|
| MobileViT-XXS | 1.3 M      | 69.0  | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt) |
| MobileViT-XS  | 2.3 M      | 74.7  | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt)  |
| MobileViT-S   | 5.6 M      | 78.3  | [Link](https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt)   |

## Training Parameter description

| Parameter            | Default          | Description                             |
|:---------------------|:-----------------|:----------------------------------------|
| device_target        | GPU              | Device type                             |
| data_url             | None             | Path of data file                       |
| num_parallel_workers | 8                | Number of parallel workers              |
| batch_size           | 64               | Number of batch size                    |
| num_classes          | 1001             | Number of classification                |
| momentum             | 0.9              | Momentum for the moving average         |
| epoch_size           | 180              | Number of epochs                        |
| keep_checkpoint_max  | 40               | Max number of checkpoint files          |
| ckpt_save_dir        | ./mobilevit.ckpt | Location of training outputs            |
| run_distribute       | True             | Distributed parallel training           |
| model_type           | xx_small         | Type of model to train                  |
| decay_epoc           | 150              | Number of decay epochs                  |
| max_lr               | 0.1              | Number of the maximum learning rate     |
| min_lr               | 1e-5             | Number of the minimum learning rate     |
| resize               | 256              | Resize the height and weight of picture |
| weight_decay         | 4e-5             | Momentum for the moving average         |

## Example

Here, how to use MobileViT model will be introduced as following.

### Dataset

First, import the relevant modules, configure the relevant hyperparameters and read the data set.

This part of the code has an API in the MindSpore Vision suite that can be called directly.

For details, please refer to the following link: https://www.mindspore.cn/vision/docs/zh-CN/r0.1/index.html.

The full ImageNet dataset can be downloaded at http://image-net.org.

You can unzip the dataset files into this directory structure and read them by MindSpore Vision's API.

```text
.dataset/
├── train/  (1000 directories and 1281167 images)
│  ├── n04347754/
│  │   ├── 000001.jpg
│  │   ├── 000002.jpg
│  │   └── ....
│  └── n04347756/
│      ├── 000001.jpg
│      ├── 000002.jpg
│      └── ....
└── val/   (1000 directories and 50000 images)
│   ├── n04347754/
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ....
│   └── n04347756/
│        ├── 000001.jpg
│        ├── 000002.jpg
│        └── ....
└──

```

### Data augmentation and Train Model

Training data augmentation methods (RandomResizedCrop --> RandomHorizontalFlip --> ToTensor).

Run the train.py to start to train the model. Through the model_type parameter, you can choose which model you want to train.

Attention, you can set that  you want to do distribute parallel training by setting the **run_distribute** parameter in train.py.

Attention, when you change the model_type, you have to change **ckpt_save_dir**, **model_type**the two parameter.

```shell

python train.py --data_url "./dataset" --epoch_size 180 --model_type "xx_small"
```

output:

```text

Epoch:[128/150], step:[1218/20018], loss:[2.871/3.098], time:78.808ms, lr:0.00522
Epoch:[128/150], step:[1219/20018], loss:[2.896/3.098], time:75.547ms, lr:0.00522
Epoch:[128/150], step:[1220/20018], loss:[2.832/3.098], time:78.639ms, lr:0.00522
Epoch:[128/150], step:[1221/20018], loss:[2.887/3.098], time:77.040ms, lr:0.00522
......
```

### Evaluate Model

After training, you can use test set to evaluate the performance of your model.
Run eval.py to achieve this. The usage of model_type parameter is same as training process.

```shell

python eval.py --model_type "xx_small" --pretrained_model "mobilevit_xxs.ckpt"
```

output:

```text
{'Top_1_Accuracy':0.62184, 'Top_5_Accuracy':0.84292}
```

### Infer

At last, Use any image from the eval dataset or the train dataset in the dataset.
Put your image in the infer folder, then run infer.py to do inference.

```shell

python MobuileViT_infer.py --model_type "xx_small" --pretrained_model "mobilevit_xxs.ckpt"
```

output:

```text
{8,'hen'}
```