# MLP-Mixer

The MLP-Mixer[1] architecture (or “Mixer” for short) is an image architecture
that doesn't use convolutions or self-attention.
Instead, Mixer’s architecture is based entirely on multi-layer perceptrons (MLPs) that
are repeatedly applied across either spatial locations or feature channels.
Mixer relies only on basic matrix multiplication routines,
changes to data layout (reshapes and transpositions), and scalar nonlinearities.

It accepts a sequence of linearly projected image patches
(also referred to as tokens) shaped as
a “patches × channels” table as an input, and maintains this dimensionality.
Mixer makes use of two types of MLP layers: channel-mixing MLPs and token-mixing MLPs.
The channel-mixing MLPs allow communication between different channels;
they operate on each token independently and take individual rows of the table as inputs.
The token-mixing MLPs allow communication between different spatial locations (tokens);
they operate on each channel independently and take individual columns of the table as inputs.
These two types of layers are interleaved to enable interaction of both input dimensions.

## Pretrained model

We use the models with GSAM[2] on ImageNet to fine-tune Cifar10. Here are the results:

| model        | 'Top_1_Accuracy           | Top_5_Accuracy  | ckpt |
| :-------------: |:-------------:| :-----:| :-----:|
| Mixer-B_16      |  0.9606 | 0.9985 | [ckpt](https://download.mindspore.cn/vision/mlpmixer/gsam_Mixer-B_16.ckpt) |
| Mixer-B_32      |  0.9364 | 0.9933 | [ckpt](https://download.mindspore.cn/vision/mlpmixer/gsam_Mixer-B_32.ckpt) |
| Mixer-S_16      |  0.9606 | 0.9985 | [ckpt](https://download.mindspore.cn/vision/mlpmixer/gsam_Mixer-S_16.ckpt) |
| Mixer-S_32      |  0.9175 | 0.9908 | [ckpt](https://download.mindspore.cn/vision/mlpmixer/gsam_Mixer-S_32.ckpt) |
| Mixer-S_8      |  0.9291 | 0.9908 | [ckpt](https://download.mindspore.cn/vision/mlpmixer/gsam_Mixer-S_8.ckpt) |

## Example

Here, how to use MLP-Mixer model will be introduce as following.

### data path

Please make sure your dataset path has the following structure.

```text
./data/
├── cifar-10-batches-py
│   ├── batches.meta
│   ├── data_batch_1
│   ├── data_batch_2
│   ├── data_batch_3
│   ├── data_batch_4
│   ├── data_batch_5
│   ├── readme.html
│   └── test_batch
└── cifar-10-python.tar.gz
```

### Train

* The following configuration for finetune.

```shell
python train.py --finetune True --model Mixer_B_16 --data_url ./data/
```

* The following configuration for train from scratch.

```shell
python train.py --finetune False --model Mixer_B_16 --data_url ./data/
```

### Evaluate

* The following configuration for evaluate finetuned model.

```shell
python eval.py --finetune True --model Mixer_B_16 --data_url ./data/
```

* The following configuration for evaluate local model.

```shell
python eval.py --finetune False --model Mixer_B_16 --data_url ./data/
```

## References

In this repository release models from the papers

* [1]. Tolstikhin I O, Houlsby N, Kolesnikov A, et al. Mlp-mixer: An all-mlp architecture for vision[J]. Advances in Neural Information Processing Systems, 2021, 34: 24261-24272.
* [2]. Zhuang J, Gong B, Yuan L, et al. Surrogate gap minimization improves sharpness-aware training[J]. arXiv preprint arXiv:2203.08065, 2022.