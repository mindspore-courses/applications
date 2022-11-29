# Contextual Residual Aggregation(CRA)

Traditional image inpainting methods can only deal with low resolution input images, while simple upsampling of low resolution inpainting results will only produce large and fuzzy results. Adding a high frequency residual image onto the large blurry image can generate a sharp result, rich in details and textures. CRA adds context aggregation residuals to the upsampled neural network inpainting results to output the final results. Through the Attention Transfer Module (ATM), the aggregation residual in the mask area is calculated from the context residual and the attention score. By building a generation countermeasure network to predict low resolution images, the cost of memory and computing time is well suppressed, and the ultra-high resolution image recovery can be more effective and high-quality.

## Pretrained model

Model trained by MindSpore:

|  dataset  |  ckpt  |
| :-----: | ------ |
| places | [ckpt](https://download.mindspore.cn/vision/cra/cra.ckpt) |

## Training Parameter description

| Parameter | Default | Description |
|:----:|:--------:|:-------:|
| image_dir | ../places | Image path of training input data |
| mask_template_dir | ../mask_template | Mask path of training input data |
| save_folder |../ckpt_out| File path of stored checkpoint file in training |
| device_target | GPU | Training device |
| device_id | 0 | Get device id |
| device_num | 1 | Get device num |
| IMG_SHAPE | [512, 512, 3] | Required dimensions of the network input tensor |
| attention_type | SOFT | compute attention type |
| coarse_alpha | 1.2 | Proportion of coarse output in loss calculation |
| gan_with_mask | False | Whether to concat mask when calculating adversarial loss |
| gan_loss_alpha | 0.001 | Proportion of adversarial loss of generator |
| in_hole_alpha | 1.2 | The influence of the generation results in the mask area on the loss value |
| context_alpha | 1.2 | The influence of the generation results outside the mask area on the loss value |
| wgan_gp_lambda | 10 | The influence of WGAN-GP loss on discriminator loss value |
| learning_rate | 1e-4 | Initial learning rate |
| lr_decrease_epoch | 2 | Number of epochs to decay over |
| lr_decrease_factor | 0.5 | The decay rate |
| run_distribute | False | Whether to run distribute |
| train_batchsize | 4 | Batch size for training |
| epochs | 15 | Epoch number for training |
| dis_iter | 1 | Train once generator when training dis_iter times discriminator |

## Example

Here, how to use CRA model will be introduec as following.

### Dataset

At first, you should download dataset by yourself. Places2 dataset is supported.

Attention, for Places2 dataset, you need to download the High resolution images training dataset, which has 443 scene categories,
including more than 1.8 million pictures of 1024 * 1024.

In addition, mask data and test data have been provided in our work.

The download link is as follows:

Places2: http://places2.csail.mit.edu/download.html.

mask_templates: https://github.com/duxingren14/Hifill-tensorflow/tree/master/mask_templates.

test: https://github.com/duxingren14/Hifill-tensorflow/tree/master/data/test.

After you get the dataset, make sure your path is as following:

```text
  CRA
   ├── places
           ├── a
               ├── auto_showroom
                         ├── 00000001.jpg
                         ├── 00000002.jpg
                         ├── 00000003.jpg
                         └── ......
               ├── auto_factory
               ├── ......
               ├── airplane_cabin
               └── airfield
           ├── b
           ├── c
           ├── ......
           ├── y
           └── z
    ├── mask_templates
           ├── 0.png
           ├── ......
           └── 99.png
    └── test
           ├──images
              ├── 0.png
              └── 1.png
           └──masks
              ├── 0.png
              └── 1.png
```

### Train

The following configuration uses 1 GPUs for training. The trained for 15 epochs, and the batch size 4.

```shell
python train.py --image_dir ../places --mask_template_dir ../mask_templates --save_folder ../ckpt_out --device_target GPU --device_id 0 --device_num 1 --run_distribute False --train_batchsize 4 --epochs 15
```

The following configurations are distributed parallel training for eight GPU cards.

```shell
mpirun -n 8 python train.py --image_dir ../places --mask_template_dir ../mask_templates --save_folder ../ckpt_out --device_target GPU --device_id 0 --device_num 8 --run_distribute True --train_batchsize 4 --epochs 15
```

output:

```text
epoch1/15, batch1/56358, d_loss is 1091.4999, g_loss is 1.3412, time is 0.5120
epoch1/15, batch1/56358, d_loss is 1238.4945, g_loss is 1.6735, time is 0.5127
epoch1/15, batch1/56358, d_loss is 1082.4247, g_loss is 1.8266, time is 0.5117
epoch1/15, batch1/56358, d_loss is 971.5017, g_loss is 1.8454, time is 0.5126
epoch1/15, batch1/56358, d_loss is 1157.3241, g_loss is 1.7420, time is 0.5127
epoch1/15, batch1/56358, d_loss is 1068.8934, g_loss is 1.5067, time is 0.5129
epoch1/15, batch1/56358, d_loss is 1284.8508, g_loss is 1.8697, time is 0.5120
epoch1/15, batch2/56358, d_loss is 987.3273, g_loss is 1.5855, time is 0.5125
epoch1/15, batch2/56358, d_loss is 1002.3116, g_loss is 1.6405, time is 0.4966
epoch1/15, batch2/56358, d_loss is 937.8546, g_loss is 1.3261, time is 0.4965
epoch1/15, batch2/56358, d_loss is 1288.6157, g_loss is 1.6953, time is 0.4973
epoch1/15, batch2/56358, d_loss is 1130.4807, g_loss is 1.6920, time is 0.4969
epoch1/15, batch2/56358, d_loss is 1203.1342, g_loss is 1.4811, time is 0.4973
epoch1/15, batch2/56358, d_loss is 1124.6455, g_loss is 1.4844, time is 0.4966
epoch1/15, batch2/56358, d_loss is 983.5717, g_loss is 1.3907, time is 0.4972
···
```

The following program takes Ascend: 8 * Ascend-910(32GB) | ARM: 192 核 768GB environment to train the places2 dataset as an example, and run the training as follows.

```shell
python train.py --image_dir ../places --mask_template_dir ../mask_templates --save_folder ../ckpt_out --device_target Ascend --device_id 0 --device_num 8 --run_distribute True --train_batchsize 4 --epochs 15
```

### Infer

The following configuration be used to infer.

```shell
python test.py --image_dir ../test/images --mask_dir ../test/masks --output_dir ../output --checkpoint_dir ../ckpt_out/generator_epoch15_batch56358.ckpt
```

#### Result

![1.jpg](attachment:1.jpg)

![0.jpg](attachment:0.jpg)
