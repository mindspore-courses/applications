# ConvNext

***

In January 2022, Facebook AI research and UC Berkeley jointly published an article called a convnet for the 2020s. In this article, convnext pure convolutional neural network is proposed, Compared with the current popular swing transformer, a series of experiments show that convnext achieves higher reasoning speed and better accuracy under the same flops.

The architectural definition of network refers to the following papers:

[1] Liu Z , Mao H , Wu C Y , et al. [A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545v2.pdf). 2022.

## Pretrained model

Model trained by MindSpore, there are 1 ckpt file.

|  model  |  ckpt  |
| ------- | ------ |
| convnext | [ckpt](https://download.mindspore.cn/vision/convnext/convnext_tiny0-300_533.ckpt) |

***

## Dataset

We will use the [ImageNet dataset](https://image-net.org/), which has a total of 1000 classes, each of which is a 224 by 224 color image. There are 1,281,167 images in the training set and 50,000 images in the test set.

After downloading dataset, place it in src/data, and unzip it. The structure of folder is as follows:

```markdown
.ImageNet/
    ├── ILSVRC2012_devkit_t12.tar.gz
    ├── train/
    ├── infer/
    └── val
```

## Training Parameter description

| Parameter | Default | Description |
|:-----|:---------|:--------|
| model | convnext_tiny | Which model you want to train |
| data_url | /home/ma-user/work/imagenet2012 | Location of data |
| num_classes | 1000 | Number of classification |
| mix_up | 0.8 | MixUp data enhancement parameters |
| cutmix | 1.0 | CutMix data enhancement parameters |
| auto_augment | rand-m9-mstd0.5-inc1 | AutoAugment parameters |
| interpolation | bicubic | Image scaling interpolation method |
| re_prob | 0 | The probability of RandomErasing |
| re_mode | pixel | The mode of RandomErasing |
| re_count | 1 | The repeat number of RandomErasing |
| mixup_prob | 1. | The probability of MixUp |
| switch_prob | 0.5 | MixUp and CutMix switching probability |
| mixup_mode | batch | The mode of MixUp |
| optimizer | adamW | Optimizer Category |
| base_lr | 0.0005 | Basic learning rate |
| warmup_lr | 0.00000007 | Learning rate Warm-up initial learning rate |
| min_lr | 0.000006 | Minimum learning rate |
| lr_scheduler | cosine_lr | Learning rate decay policy |
| warmup_length | 20 | Learning rate Warm-up rounds |
| nonlinearity | GELU | Activation function category |
| image_size | 224 | Image size |
| amp_level | O1 | Mixed precision strategy |
| beta | [ 0.9, 0.999 ] | The parameters of the adamw |
| clip_global_norm_value | 5. | Global gradient norm clipping threshold |
| is_dynamic_loss_scale | True | Whether to use dynamic scaling |
| epochs | 300 | Number of train epoch |
| label_smoothing | 0.1 | Label Smoothing Parameters |
| weight_decay | 0.05 | Weight decay parameter |
| batch_size | 300 | Batch Size |
| with_ema | False | Whether to use ema updates |
| ema_decay | 0.9999 | ema mobility coefficient |
| num_parallel_workers | 16 | Number of data preprocessing threads |
| device_target | Ascend | GPU or Ascend |

## Performance

| model | Resource | Speed | Total time |
|:------- |:---------|:------|:-----------|
| convnext_tiny |Ascend 910|8pc(Ascend): 570.730ms/step|8pc(Ascend): 31h38m|

## Examples

***

### Train

- The following configuration uses 1 Ascend for training.

  ```shell
  python train.py --data_url /home/ma-user/work/imagenet2012
  ```

  output:

  ```text
    epoch: 1 step: 4270, loss is 6.669617652893066
    epoch time: 2986602.286 ms, per step time: 699.438 ms
    epoch: 2 step: 4270, loss is 6.4347357749938965
    epoch time: 2781168.379 ms, per step time: 651.327 ms
    epoch: 3 step: 4270, loss is 5.996033668518066
    epoch time: 2673875.158 ms, per step time: 626.200 ms
    epoch: 4 step: 4270, loss is 5.784404277801514
    epoch time: 2648504.227 ms, per step time: 620.259 ms
  ...
  ```

- The following configuration uses 8 Ascends for training.

  ```shell
  bash run_distribute_train_ascend.sh /home/ma-user/work/imagenet2012
  ```

  output:

  ```text
    epoch: 1 step: 533, loss is 6.813085556030273
    epoch time: 1347728.492 ms, per step time: 2528.571 ms
    epoch: 2 step: 533, loss is 6.64223575592041
    epoch time: 302692.096 ms, per step time: 567.903 ms
    epoch: 3 step: 533, loss is 6.211109161376953
    epoch time: 300695.537 ms, per step time: 564.157 ms
    epoch: 4 step: 533, loss is 5.540595054626465
    epoch time: 301485.360 ms, per step time: 565.639 ms
    epoch: 5 step: 533, loss is 6.099679470062256
    epoch time: 301652.908 ms, per step time: 565.953 ms
    epoch: 6 step: 533, loss is 5.2530517578125
    epoch time: 301642.222 ms, per step time: 565.933 ms
    epoch: 7 step: 533, loss is 4.9563398361206055
    epoch time: 301513.162 ms, per step time: 565.691 ms
  ...
  ```

### Eval

- The following configuration uses 1 Ascend for Eval.

  ```shell
  python eval.py --data_url /home/ma-user/work/imagenet2012 --pretrained /home/ma-user/work/course/ckpt_0/convnext_tiny0-10_10.ckpt
  ```

  output:

  ```text
    {'Top1-Acc': 0.8230522088353414, 'Top5-Acc': 0.9593172690763052}
  ```  

### infer

- The following configuration for infer.

  ```shell
  python infer.py --data_url /home/ma-user/work/imagenet2012 --pretrained /home/ma-user/work/course/ckpt_0/convnext_tiny0-10_10.ckpt
  ```

  output:

  ```text
  {394: 'sturgeon'}
  ```  

  result:
  <div align=center><img src="./images/convnext_infer.jpg"></div>