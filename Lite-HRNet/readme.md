# Lite-HRNet

The Lite-HRNet(Lite-HighResolutionNetwork) is a light weight backbone for high-resolution dependent computer vision tasks like Semantic Segmentation and Pose Estimation. It is a lightweight variant of HRNet(HighResolutionalNetwork), in which the authors replaces 1*1 convolutional blocks in HRNet with their innovative Cross Channel Weighting(CCW) to reduce both the time and spatial complexity.
In this case, we implement the Lite-HRNet using Mindspore for the task of human pose estimation.

## Performance and Pretrained Model

On COCO2017:

| Arch  | Input Size | #Params | FLOPs | AP | AP<sup>50</sup> | AP<sup>75</sup> | AR | AR<sup>50</sup> | ckpt |
| :----------------- | :-----------: | :------: | :-----------: | :------: |:------: | :------: | :------: | :------: | :------: |
| Naive Lite-HRNet-18  | 256x192 | 0.7M | 194.8M | 0.608 | 0.873 | 0.683 | 0.645 | 0.886 | [Mindspore Download](https://download.mindspore.cn/vision/litehrnet/ckpts/naive_litehrnet_18_coco_256x192.ckpt)|
| Wider Naive Lite-HRNet-18  | 256x192 | 1.3M | 311.1M | 0.643 | 0.893 | 0.725 | 0.680 | 0.901 | [Mindspore Download](https://download.mindspore.cn/vision/litehrnet/ckpts/wider_naive_litehrnet_18_coco_256x192.ckpt) |
| Lite-HRNet-18 | 256x192 | 1.1M | 205.2M |0.624 | 0.884 | 0.703 | 0.663 | 0.895 | [Mindspore Download](https://download.mindspore.cn/vision/litehrnet/ckpts/litehrnet_18_coco_256x192.ckpt) |
| Lite-HRNet-18  | 384x288 | 1.1M | 461.6M | 0.668 | 0.895 | 0.739 | 0.702 | 0.909 | [Mindspore Download](https://download.mindspore.cn/vision/litehrnet/ckpts/litehrnet_18_coco_384x288.ckpt)  |
| Lite-HRNet-30  | 256x192 | 1.8M | 319.2M | 0.655 | 0.895 | 0.737 | 0.694 | 0.908 | [Mindspore Download](https://download.mindspore.cn/vision/litehrnet/ckpts/litehrnet_30_coco_256x192.ckpt)  |
| Lite-HRNet-30  | 384x288 | 1.8M | 717.8M | 0.698 | 0.905 | 0.781 | 0.731 | 0.918 | [Mindspore Download](https://download.mindspore.cn/vision/litehrnet/ckpts/litehrnet_30_coco_384x288.ckpt) |

On MPII:

| Arch  | Input Size | #Params | FLOPs | Mean | Mean@0.1 | ckpt |
| :----------------- | :-----------: | :------: | :-----------: | :------: |:------: | :------: |
| Naive Lite-HRNet-18  | 256x192 | 0.7M | 194.8M | 0.842 | 0.270 | [Mindspore Download](https://download.mindspore.cn/vision/litehrnet/ckpts/naive_litehrnet_18_mpii_256x256.ckpt)|
| Wider Naive Lite-HRNet-18  | 256x256 | 1.3M | 311.1M | 0.859 | 0.288 | [Mindspore Download](https://download.mindspore.cn/vision/litehrnet/ckpts/wider_naive_litehrnet_18_mpii_256x256.ckpt) |
| Lite-HRNet-18 | 256x256| 1.1M | 205.2M |0.842 | 0.262 |  [Mindspore Download](https://download.mindspore.cn/vision/litehrnet/ckpts/litehrnet_18_mpii_256x256.ckpt) |
| Lite-HRNet-30  | 256x256 | 1.8M | 319.2M | 0.861 | 0.280 |  [Mindspore Download](https://download.mindspore.cn/vision/litehrnet/ckpts/litehrnet_30_mpii_256x256.ckpt)  |

## Prepare Dataset

You have to download COCO2017 dataset and annotation in [COCODataset](https://cocodataset.org) and make sure that the your path is organized as following:

```text
Lite-HRNet/
    ├── imgs
    ├── src
    ├── annotations
        ├──person_keypoints_train2017.json
        └──person_keypoints_train2017.json
    ├── train2017
    └── val2017
```

## Requirements

## Train

```python
python src/train.py
```

| Optional Args | Default Value | Explanation |
| :----------------- | :-----------: | :-----------------: |
| --model_type | lite_18 | The type of model |
| --target_res | 256x192 | Resolution of resized input images |
| --checkpoint_path | ./ckpts | Where to save or load checkpoints |
| --train_batch | 32 | Training batch size |
| --save_checkpoint_steps | 500 | The interval of saving checkpoints |
| --load_ckpt | False | Load a checkpoint and continue training |

## Evaluate

```python
python src/eval.py
```

| Optional Args | Default Value | Explanation |
| :----------------- | :-----------: | :-----------------: |
| --model_type | lite_18 | The type of model |
| --target_res | 256x192 | Resolution of resized input images |
| --checkpoint_path | ./ckpts | Where to load checkpoints |
| --output_path | ./eval_result | Where to save the evaluate result json file|

## Infer

```python
python src/eval.py
```

| Optional Args | Default Value | Explanation |
| :----------------- | :-----------: | :-----------------: |
| --model_type | lite_18 | The type of model |
| --target_res | 256x192 | Resolution of resized input images |
| --checkpoint_path | ./ckpts | Where to load checkpoints |
| --infer_data_root | ./infer_data | Where to load the input data |
| --out_data_root | ./out_data | Where to save the output data |

## Result

![图片](./imgs/man_infer.jpg)
