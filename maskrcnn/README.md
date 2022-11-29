# Mask-R-CNN

A Mindspore version of Mask R-CNN, trained on GPU. The backbones include Resnet50 and Mobilenetv1

Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks on each Region of Interest (RoI), in parallel with the existing branch for classification and bounding box regression.

The mask branch is a small FCN applied to each RoI, predicting a segmentation mask in a pixel-topixel manner.

Mask R-CNN is simple to implement and train given the Faster R-CNN framework, which facilitates a wide range of flexible architecture designs.

Additionally, the mask branch only adds a small computational overhead, enabling a fast system and rapid experimentation.

![image1](images/framework.png)

Without bells and whistles, Mask R-CNN surpasses all previous state-of-the-art single-model results on the COCO instance segmentation task, including the heavilyengineered entries from the 2016 competition winner in 2018.

## Model Features

1. Mask R-CNN adds a branch of Mask prediction on the basis of Faster R-CNN

2. Mask R-CNN proposes ROI ALign.

## Pretrained Model

|  model | training dataset | device target | backbone | $AP$ | $AP_{50}$ | ckpt |
|  ----  | ----  | ---- | ---- | ---- | ---- | ---- |
| maskrcnn_coco2017_bbox37.4_segm32.9 | coco2017 | GPU | ResNet50+FPN | 0.374 | 0.599 | [model](https://download.mindspore.cn/vision/maskrcnn/checkpoint/maskrcnn_coco2017_acc32.9.ckpt) |
| maskrcnnmobilenetv1_coco2017_bbox22.2_segm15.8 | coco2017 | GPU | MobileNetV1+FPN | 0.222 | 0.401 | [model](https://download.mindspore.cn/vision/maskrcnn/checkpoint/maskrcnnmobilenetv1_coco2017_bbox24.00_segm21.5.ckpt) |
| mask_rcnn_coco.h5 | coco2017 | GPU | ResNet50+FPN | 0.320 | 0.581 | [model](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5) |

## Training Parameter Description

Here, we list some important parameters for training. Moreover, you can check the configuration files for details.
|  Parameter | Default | Description |
|  ----  | ----  | ---- |
| workers  | 1 | Number of parallel workers |
| device_target  | GPU | Device type |
| learning_rate  | 0.002 | learning rate |
| weight_decay  | 1e-4 | Control weight decay speed |
| total_epoch  | 13 | Number of epoch |
| batch_size | 2 | Batch size |
| dataset  | coco | Dataset name |
| pre_trained  | ./checkpoint | The path of pretrained model |
| checkpoint_path  | ./ckpt_0 | The path to save |

## Example

Here, how to use Mask R-CNN and Mask R-CNN MobileNetV1 will be introduced as follow.

Taking Mask R-CNN for example, Mask R-CNN MobileNetV1 follows the same procedure.

### Dataset

At first, you should download [coco2017](https://cocodataset.org/#download) dataset by yourself.

Simply, you can use this command directly to download coco2017 as well.

```powershell
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
```

Then you can unzip them to a specified folder.

When you finish download of coco2017, you need change "coco_root" and other parameters in condiguration files.

After you get the dataset, make sure your path is as following:

```powershell
.
└─cocodataset
  ├─annotations
    ├─instance_train2017.json
    └─instance_val2017.json
  ├─val2017
  └─train2017
```

 If you'd like to use other datasets, please change "dataset" to "other" in the configuration when you run the script.

### Data Augmentation

Before you start to train the model. Data augmentation is necessary for your dataset and create train data and test data. For coco dataset, you can use dataset.py to add masks to images and convert them to MindRecord. MindRecord is a specified data format, which optimize the performance of MindSpore in some scenarios.

### Train Model

When you should change parameters and dataset, you can change parameters and dataset path in config.py.

```powershell
python train.py
```

### Evaluate Model

After training, you can use validation set to evaluate the performance of your model.

Run eval.py to achieve this. The usage of model_type parameter is same as training process.

```powershell
python eval.py
```

### Evaluation Results

#### Mask R-CNN ResNet50

##### Model trained by MindSpore on GPU

Evaluate annotation type *bbox*
| IoU | area | maxDets | Average Precision (AP) |
|---|---|---|---|
| 0.50:0.95 | all | 100 | 0.374|
| 0.50      | all | 100 | 0.599|
| 0.75      | all | 100 | 0.403|
| 0.50:0.95 | small | 100 | 0.235|
| 0.50:0.95 | medium | 100 | 0.415|
| 0.50:0.95 | large | 100 | 0.474|
| 0.50:0.95 | all | 1 | 0.312|
| 0.50:0.95 | all | 10 | 0.501|
| 0.50:0.95 | all | 100 | 0.530|
| 0.50:0.95 | small | 100 | 0.363|
| 0.50:0.95 | medium | 100 | 0.571|
| 0.50:0.95 | large | 100 | 0.656|

Evaluate annotation type *segm*
| IoU | area | maxDets | Average Precision (AP) |
|---|---|---|---|
| 0.50:0.95 | all | 100 | 0.329|
| 0.50      | all | 100 | 0.555|
| 0.75      | all | 100 | 0.344|
| 0.50:0.95 | small | 100 | 0.165|
| 0.50:0.95 | medium | 100 | 0.357|
| 0.50:0.95 | large | 100 | 0.477|
| 0.50:0.95 | all | 1 | 0.284|
| 0.50:0.95 | all | 10 | 0.436|
| 0.50:0.95 | all | 100 | 0.455|
| 0.50:0.95 | small | 100 | 0.283|
| 0.50:0.95 | medium | 100 | 0.490|
| 0.50:0.95 | large | 100 | 0.592|

#### Mask R-CNN Mobilenetv1

##### Model trained by MindSpore on GPU

Evaluate annotation type *bbox*
| IoU | area | maxDets | Average Precision (AP) |
|---|---|---|---|
| 0.50:0.95 | all | 100 | 0.235|
| 0.50      | all | 100 | 0.409|
| 0.75      | all | 100 | 0.241|
| 0.50:0.95 | small | 100 | 0.145|
| 0.50:0.95 | medium | 100 | 0.250|
| 0.50:0.95 | large | 100 | 0.296|
| 0.50:0.95 | all | 1 | 0.243|
| 0.50:0.95 | all | 10 | 0.397|
| 0.50:0.95 | all | 100 | 0.418|
| 0.50:0.95 | small | 100 | 0.264|
| 0.50:0.95 | medium | 100 | 0.449|
| 0.50:0.95 | large | 100 | 0.515|

Evaluate annotation type *segm*
| IoU | area | maxDets | Average Precision (AP) |
|---|---|---|---|
| 0.50:0.95 | all | 100 | 0.191|
| 0.50      | all | 100 | 0.350|
| 0.75      | all | 100 | 0.190|
| 0.50:0.95 | small | 100 | 0.095|
| 0.50:0.95 | medium | 100 | 0.204|
| 0.50:0.95 | large | 100 | 0.278|
| 0.50:0.95 | all | 1 | 0.206|
| 0.50:0.95 | all | 10 | 0.315|
| 0.50:0.95 | all | 100 | 0.328|
| 0.50:0.95 | small | 100 | 0.194|
| 0.50:0.95 | medium | 100 | 0.350|
| 0.50:0.95 | large | 100 | 0.424|

## Inference

At last, you can use your own images to test the trained model. Put your image in the images folder, then run infer.py to do inference.

```powershell
python infer.py
```

![image2](images/infer.png)

## Notes

1. You can fine-tune the model and change checkpoint in config.py when change the backbone.

2. You can create your own dataset following dataset.py, and change dataset root in config.py.

## Reference

[1] He K, Gkioxari G, Dollár P, et al. Mask r-cnn[C]//Proceedings of the IEEE international conference on computer vision. 2017: 2961-2969.
