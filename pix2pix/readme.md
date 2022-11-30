# Pix2Pix

Pix2Pix is a deep learning image conversion model based on conditional generative adversarial network. The model can realize semantic or label to real image, grayscale image to color image, aerial image to map, day to night, line drawing to physical image conversion. Pix2Pix is a general framework for image translation based on conditional GAN, which realizes the generalization of model structure and loss function, and has achieved remarkable results on many image translation datasets.In Pix2Pix, the generator and the discriminator play against each other to optimize the two models at the same time.

## Pretrained model

Model trained by MindSpore, Six datasets correspond to six ckpt files.

|  dataset  |  ckpt  |
| :-----: | ------ |
| maps | [ckpt](https://download.mindspore.cn/vision/pix2pix/maps/Generator_200.ckpt) |
| cityscapes |[ckpt](https://download.mindspore.cn/vision/pix2pix/cityscapes/Generator_200.ckpt) |
| facades |[ckpt](https://download.mindspore.cn/vision/pix2pix/facades/Generator_200.ckpt) |
| night2day |[ckpt](https://download.mindspore.cn/vision/pix2pix/night2day/Generator_17.ckpt) |
| edges2shoes |[ckpt](https://download.mindspore.cn/vision/pix2pix/edge2shoes/Generator_15.ckpt) |
| edges2handbags |[ckpt](https://download.mindspore.cn/vision/pix2pix/edge2handbags/Generator_15.ckpt) |

## Training Parameter description

| Parameter | Default | Description |
|:----:|:--------:|:-------:|
| device_target | GPU | Device id of GPU or Ascend |
| epoch_num | 200 | Epoch number for training |
| batch_size | 1 | Batch size |
| beta1 | 0.5 | Adam beta1 |
| beta2 | 0.999 | Adam beta2 |
| load_size | 286 | Scale images to this size |
| train_pic_size | 256 | Train image size |
| val_pic_size | 256 | Eval image size |
| lambda_dis | 0.5 | Weight for discriminator loss |
| lambda_gan | 0.5 | Weight for GAN loss |
| lambda_l1 | 100 | Weight for L1 loss |
| lr | 0.0002 | Initial learning rate |
| n_epochs | 256 | Number of epochs with the initial learning rate |
| g_in_planes | 3 | The number of channels in input images |
| g_out_planes | 3 | The number of channels in output images |
| g_ngf | 64 | The number of filters in the last conv layer |
| g_layers | 8 | The number of downsamplings in UNet |
| d_in_planes | 6 | Input channel |
| d_ndf | 64 | The number of filters in the last conv layer |
| d_layers | 3 | The number of ConvNormRelu blocks |
| alpha | 0.2 | LeakyRelu slope |
| init_gain | 0.02 | Scaling factor for normal xavier and orthogonal |
| train_fakeimg_dir | results/fake_img/ | File path of stored fake img in training |
| loss_show_dir | results/loss_show | File path of stored loss img in training |
| ckpt_dir | results/ckpt | File path of stored checkpoint file in training |
| ckpt | results/ckpt/Generator_200.ckpt | File path of checking point file used in validation |
| predict_dir | results/predict/ | File path of generated image in validation |

The above table takes the facades dataset as an example. The corresponding parameters of different data sets are different. The reference values are shown as follows.

| Datasets and Parameters | epoch_num | batch_size | dataset_size | ckpt |
|:----:|:--------:|:-------:|:-------:|:-------:|
| facades | 200 | 1 | 400 | 'results/ckpt/Generator_200.ckpt' |
| cityscapes | 200 | 1 | 2975 | 'results/ckpt/Generator_200.ckpt' |
| maps | 200 | 1 | 1096 | 'results/ckpt/Generator_200.ckpt' |
| night2day | 17 | 4 | 17823 | 'results/ckpt/Generator_17.ckpt' |
| edges2shoes | 15 | 4 | 49825 | 'results/ckpt/Generator_15.ckpt' |
| edges2handbags | 15 | 4 | 138567 | 'results/ckpt/Generator_15.ckpt' |

## Example

Here, how to use Pix2Pix model will be introduec as following.

***

### Train

- The following configuration uses 1 GPUs for training. We select edges2handbags.tar.gz. The trained for  15 epochs, and the batch size 4.

```shell
  python train.py --device_target GPU --train_data_dir /home/pix2pix/end/datasets/edges2handbags/train/ --epoch_num 15 --batch_size 4 --dataset_size 138567 --val_data_dir /home/pix2pix/end/datasets/edges2handbags/val/ --ckpt results/ckpt/Generator_15.ckpt --train_fakeimg_dir results/fake_img/ --loss_show_dir results/loss_show --ckpt_dir results/ckpt
```

output:

```text
ms per step : 573.098  epoch:  1 / 15  step:  0 / 34641  Dloss:  0.8268157  Gloss:  96.03406
ms per step : 48.086  epoch:  1 / 15  step:  100 / 34641  Dloss:  0.08200404  Gloss:  14.9787245
ms per step : 49.513  epoch:  1 / 15  step:  200 / 34641  Dloss:  1.7021327  Gloss:  11.134652
···
ms per step : 48.005  epoch:  15 / 15  step:  34400 / 34641  Dloss:  6.0905662e-05  Gloss:  13.91369
ms per step : 48.055  epoch:  15 / 15  step:  34500 / 34641  Dloss:  6.202688e-05  Gloss:  17.475426
ms per step : 47.031  epoch:  15 / 15  step:  34600 / 34641  Dloss:  1.3251489e-06  Gloss:  17.247965
```

The same program can also run distributed parallel training. The following program takes Ascend: 4*Ascend-910(32GB) | ARM: 96 cores 256GB environment to train the facedes dataset as an example, and run the training as follows.

```shell
  python train.py --device_target Ascend --run_distribute True --train_data_dir /home/ma-user/modelarts/inputs/train_data_dir_0
```

output:

```text
{
    "status": "completed",
    "group_count": "1",
    "group_list": [
        {
            "group_name": "worker",
            "device_count": "4",
            "instance_count": "1",
            "instance_list": [
                {
                    "pod_name": "ma-job-937e0416-02fd-47e0-8997-7890009fdb5c-worker-0",
                    "server_id": "192.168.0.99",
                    "devices": [
                        {
                            "device_id": "4",
                            "device_ip": "192.1.42.193"
                        },
                        {
                            "device_id": "5",
                            "device_ip": "192.2.43.229"
                        },
                        {
                            "device_id": "6",
                            "device_ip": "192.3.146.124"
                        },
                        {
                            "device_id": "7",
                            "device_ip": "192.4.206.128"
                        }
                    ]
                }
            ]
        }
    ]
}
Device ID : 3
ms per step : 11.199 epoch:  1 / 200 step:  100 / 400 Dloss:  0.227907 Gloss:  27.540344
Device ID : 0
ms per step : 739.631 epoch:  1 / 200 step:  0 / 400 Dloss:  0.8247761 Gloss:  41.588917
Device ID : 3
ms per step : 12.217 epoch:  1 / 200 step:  200 / 400 Dloss:  0.028766029 Gloss:  42.01359
Device ID : 1
ms per step : 472.742 epoch:  1 / 200 step:  0 / 400 Dloss:  0.95506287 Gloss:  43.79847
...
Device ID : 2
ms per step : 10.23 epoch:  200 / 200 step:  200 / 400 Dloss:  0.20082012 Gloss:  16.692669
Device ID : 2
ms per step : 9.693 epoch:  200 / 200 step:  300 / 400 Dloss:  0.038278762 Gloss:  22.89386
image generated from epoch 200 saved
The learning rate at this point is： 4e-06
epoch 200 saved
epoch 200 D&G_Losses saved
epoch 200 finished
ckpt generated from epoch 200 saved
```

### Infer

- The following configuration continues to be used to infer the edges2handbags dataset.

```shell
python eval.py --device_target GPU --train_data_dir /home/pix2pix/end/datasets/edges2handbags/train/ --epoch_num 15 --batch_size 4 --dataset_size 138567 --val_data_dir /home/pix2pix/end/datasets/edges2handbags/val/ --ckpt results/ckpt/Generator_15.ckpt
```

**Result**

![results2.png](./images/results2.png)