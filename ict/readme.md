# ICT

Image completion  aims to fill in the missing parts of an image with visually realistic and semantically appropriate content, and is widely used in the field of image processing. Convolutional neural networks have made great progress in the field of computer vision due to their powerful texture modeling capabilities. However, convolutional neural networks do not perform well in understanding the global structure. The development of transformers in recent years has proved its ability to model long-term relationships, but the computational complexity of the transformer hinders its application in processing high-resolution images. ICT combines the advantages of these two methods into the image completion task. First, the transformer is used to reconstruct the appearance prior, and the multivariate coherent structure and some rough textures are restored, and then the convolutional neural network is used for texture complementation, which enhances the high-resolution image. The local texture details of the rough prior guided by the rate mask image.

## Pretrained model

We tested the model on the imagenet dataset, noting that since the transformer is relatively slow, we only tested a subfolder **n02410509** of the validation set, which took about 8 minutes.

Attention, this model relies on [VGG19](https://download.mindspore.cn/vision/ict/VGG19.ckpt), the weights file of VGG19 has been extracted and available for download.

### Model trained by MindSpore

| Transofrmer |Upsample | PSNR$\uparrow$ |MAE$\downarrow$|
| ------- |----|-----|---|
| [ImageNet_best.ckpt](https://download.mindspore.cn/vision/ict/ms_train/Transformer/ImageNet_best.ckpt) |[InpaintingModel_gen_best.ckpt](https://download.mindspore.cn/vision/ict/ms_train/Upsample/InpaintingModel_gen_best.ckpt)|27.330217|0.021267721|

### Model trained by PyTorch

| Transofrmer |Upsample | PSNR$\uparrow$ |MAE$\downarrow$|
| ------- |----|-----|---|
| [ImageNet.ckpt](https://download.mindspore.cn/vision/ict/origin/Transformer/ImageNet.ckpt) |[InpaintingModel_gen.ckpt](https://download.mindspore.cn/vision/ict/origin/Upsample/ImageNet/InpaintingModel_gen.ckpt)|26.976389|0.02261999|

## Dataset

At first, you should download dataset by yourself. [ImageNet](https://image-net.org/) dataset is supported.

After you get the dataset, make sure your path is as following:

```text
# ImageNet dataset
.imagenet/
    └── train
    |    ├── n04347754
    |    |      ├── 000001.jpg
    |    |      ├── 000002.jpg
    |    |      └── ....
    |    └── n04347756
    |           ├── 000001.jpg
    |           ├── 000002.jpg
    |           └── ....
    └── val
         |      ├── n04347754
         |      ├── 000001.jpg
         |      ├── 000002.jpg
         |      └── ....
         └── n04347756
                ├── 000001.jpg
                ├── 000002.jpg
                └── ....

```

In the image completion task, we also need a mask dataset to mask the image to obtain images with damaged pixels. The [mask_dataset](https://www.dropbox.com/s/01dfayns9s0kevy/test_mask.zip?dl=0) can be downloaded by itself. After decompression, the file has the following directory structure:

```text
# Mask dataset
.mask/
├── testing_mask_dataset/
   ├── 000001.png
   ├── 000002.png
   ├── 000003.png
   └── ....
```

After downloading the weights file and dataset, your folder directory structure should look like this:

```text
.ict/
├── ckpts_ICT/                      # The weight checkpoint folder
│  ├── ms_train/
│  ├── origin/
│  └── VGG19.ckpt
├── mask/                           # Mask dataset folder
│  ├── testing_mask_dataset/
│  │   ├── 00000.png  
│  │   ├── 00001.png
│  │   ├── 00002.png
│  │   └── ....
├── Guided_Upsample/                # Second stage Upsample
├── Transformer/                    # Fitst stage Transformer
├── images/                         # Folder for displaying pictures
├── run.py                          # One stage infer or eval
├── ict.ipynb                       # Executable case file
├── kmeans_centers.npy              # Cluster center dependency file
└── readme.md
```

## Train

### Transformer Training Parameter description

| Parameter | Default | Description |
|:-----|:---------|:--------|
| data_path | | Indicate where is the training set |
| mask_path | | Indicate where is the mask |
| ckpt_path | | The path of resume ckpt |
| device_id | 0 | Device id |
| device_target | GPU | Device type |
| save_path | ./checkpoint | Save checkpoints path |
| batch_size | 2 | The number of train batch size |
| train_epoch | 5 | How many epochs |
| random_stroke | False | Use the generated mask |
| use_ImageFolder | False | Using the original folder for ImageNet dataset |
| prior_size | 32 | Input sequence length = prior_size * prior_size |
| learning_rate | 3e-4 | Value of learning rate |
| beta1 | 0.9 | Value of beta1 |
| beta2  | 0.95 | Value of beta2 |

### Train Transformer Model

Before starting to train the model, please get the dataset path **data_path** and mask dataset path **mask_path**.

When you change the dataset path, you have to change **data_path**, **mask_path** the two parameter.

Attention, if you want to modify the path, use absolute paths to reduce unnecessary errors.

Run the train.py to start to train the model. With the **ckpt_path** parameter, you can resume training from an existing model.

#### Example

```shell
python train.py --mask_path '../mask/testing_mask_dataset' --data_path '/data0/imagenet2012/train' --use_ImageFolder
```

#### Output

The following is a partial display of the training output

```text
Epoch: [0 / 5], step: [32 / 1281167], loss: 2.3844809532165527
Epoch: [0 / 5], step: [34 / 1281167], loss: 2.360657215118408
Epoch: [0 / 5], step: [36 / 1281167], loss: 2.3280274868011475
......
Epoch: [0 / 5], step: [75372 / 1281167], loss: 1.682691216468811
Epoch: [0 / 5], step: [75374 / 1281167], loss: 1.682668685913086
......
```

### Upsample Training Parameter description

| Parameter | Default | Description |
|:-----|:---------|:--------|
| input | | Indicate where is the training set |
| mask | | Path to the kmeans|
| ckpt_path | | The path of resume ckpt |
| device_id | 0 | Device id |
| device_target | GPU | Device type |
| save_path | ./checkpoint | Save checkpoints path |
| kmeans | ./kmeans_centers.npy | Path to the VGG |
| vgg_path | ./VGG19.ckpt | Indicate where is the kmeans center |
| image_size | 256 | The size of origin image |
| prior_size | 32 | The size of prior image from transformer |
| prior_random_degree | 1 | During training, how far deviate from |
| use_degradation_2 | False | Use the new degradation function |
| mode | 1 | 1 is train or 2 is test |
| mask_type | 2 | The type of mask |
| max_iteration | 25000 | How many run iteration |
| batch_size | 32 | The number of train batch size |
| D2G_lr | 0.1 | Value of discriminator/generator learning rate ratio |
| lr | 0.0001 | Value of learning rate |
| beta1 | 0.9 | Value of beta1 |
| beta2  | 0.9 | Value of beta2 |

### Train Upsample Model

Before starting to train the model, please get the dataset path **input** and mask dataset path **mask**.

When you change the dataset path, you have to change **input**, **mask** the two parameter.

Attention, if you want to modify the path, use absolute paths to reduce unnecessary errors.

Run the train.py to start to train the model. With the **ckpt_path** parameter, you can resume training from an existing model.

#### Example

```shell
python train.py --input '/data0/imagenet2012/train' --mask '../mask/testing_mask_dataset'
```

#### Output

The following is a partial display of the training output

```text
Epoch: [1], step: [0 / 40037], psnr: 15.069424, mae: 0.1978589
Epoch: [1], step: [100 / 40037], psnr: 18.389233, mae: 0.13529176
Epoch: [1], step: [200 / 40037], psnr: 19.784353, mae: 0.1134068
......
Epoch: [1], step: [24800 / 40037], psnr: 26.257063, mae: 0.038081832
Epoch: [1], step: [24900 / 40037], psnr: 26.259693, mae: 0.038054496
```

## Infer

After training, you can use testset image to test your model.

Put your image in the folder, then run **Transformer/infer.py** to generate image priors.

Attention, **ckpt_path** is the path of the trained transformer model. As before, it is recommended to use absolute paths for all paths, including image paths and mask paths.

```shell
python infer.py --ckpt_path '../ckpts_ICT/origin/Transformer/ImageNet.ckpt' --image_url '../input' --mask_url '../mask/testing_mask_dataset' --GELU_2 --save_url '../save'
```

Then we run **Guided_Upsample/infer.py** to combine the image prior information to restore the image to its original resolution.

Attention, the parameter **prior** must be the same as the **save_url** in the above command.

```shell
python infer.py --ckpt_path '../ckpts_ICT/origin/Upsample/ImageNet/InpaintingModel_gen.ckpt' --input '../input' --mask '../mask/testing_mask_dataset' --prior '../save' --save_path '../save'
```

### Output

```text
PSNR: 25.320496, MAE: 0.022787869
```

In addition, we also provide **run.py** to complete inference in one stage.

Attention, since the directory is switched in the program, please use an absolute path, using a relative path may cause run error.

```shell
python run.py --transformer_ckpt '../ckpts_ICT/origin/Transformer/ImageNet.ckpt' --upsample_ckpt '../ckpts_ICT/origin/Upsample/ImageNet/InpaintingModel_gen.ckpt' --input_image '../input' --input_mask '../mask/testing_mask_dataset' --save_place '../save'
```

**Visualize Results**

The following picture is the processed picture of the inference result, the first picture is the real picture, the second picture is the damaged picture processed by the mask, and the third picture is the output picture of the model.

![result](./images/result.png)