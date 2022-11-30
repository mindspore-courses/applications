# ECG

The electrocardiogram (ECG) is a standard test used to monitor the activity of the heart. Many cardiac abnormalities
will be manifested in the ECG including arrhythmia which is a general term that refers to an abnormal heart rhythm. The
basis of arrhythmia diagnosis is the identification of normal versus abnormal individual heart beats, and their correct
classification into different diagnoses, based on ECG morphology. Heartbeats can be sub-divided into five categories
namely non-ectopic, supraventricular ectopic, ventricular ectopic, fusion, and unknown beats. It is challenging and
time-consuming to distinguish these heartbeats on ECG as these signals are typically corrupted by noise. We developed a
9-layer deep convolutional neural network (CNN) to automatically identify 5 different categories of heartbeats in ECG
signals. When properly trained, the proposed CNN model can serve as a tool for screening of ECG to quickly identify
different types and frequency of arrhythmic heartbeats.

## Pretrained model

Model trained by MindSpore:

| model | precision | ckpt                                                               |
| --------|-----|--------------------------------------------------------------------|
| ECG_net | ... | [ckpt](https://download.mindspore.cn/vision/pfld/PFLD1X_300W.ckpt) |

Model trained by PyTorch:

| model | precision |
| ---------|-----|
| ECG_net |0.9403|

## Training Parameter description

| Parameter             | Default | Description                             |
|:----------------------|:---------|:----------------------------------------|
| graph_mode            | True | Graph_mode type                         |
| device_target         | GPU | Device type                             |
| data_path             | "/30X_eu_MLIII.csv" | Path of data                            |
| label_path            |"/30Y_eu_MLIII.csv" | Path of label                           |
| batch_size            | 128 | Number of batch size                    |
| momentum              | 0.7 | Number of momentum                      |
| lr                    | 0.003 | Number of learning rate                 |
| weight_decay          | 1e-6 | Number of weight_decay                  |
| save_checkpoint_steps | 1000 | Number of steps about saving checkpoint |
| keep_checkpoint_max   | 10 | The maximum number of saving checkpoint |
| checkpoint_dir        | "/save/" | Path of saving checkpoint               |
| per_print_times       | 100 | Print after per_print_times times       |

## Example

Here, how to use PFLD model will be introduec as following.

### Dataset

At first, you should download dataset by yourself.
[European ST-T](https://physionet.org/content/edb/1.0.0/)
、 [MIT-BIH Arrhythmia](https://physionet.org/content/mitdb/1.0.0/)
、[MIT-BIH ST Change](https://physionet.org/content/stdb/1.0.0/)
and [Sudden Cardiac Death Holter](https://physionet.org/content/sddb/1.0.0/) dataset is supported.

Attention,......
After you get the dataset, make sure your path is as following:

```text

.datasets/
    └── 300W
    |    ├── 300W_annotations
    |    |      └── Mirrors68.txt
    |    └── 300W_images
    |           ├── afw
    |           ├── helen
    |           ├── ibug
    |           └── ifpw
    └── WFLW
    |    ├── WFLW_annotations
    |    |      ├── list_98pt_rect_attr_train_test
    |    |      ├── list_98pt_test
    |    |      └── Mirrors98.txt
    |    └── WFLW_images
    |           ├── 0-Parade
    |           ├── 1-Handshaking
    |           ......
    └── infer_image

```

### Data augmentation and Train Model

Before you start to train the model. Data augmentation is necessary for your dataset and create train data and test
data.

After that you will have another two folders in you ... folder or ... folder.

Attention, ...

Run the train.py to start to train the model.

Attention, when you change the dataset, you have to change **target_dataset**, **model_type**, **train_file_path** the
three parameter.

```shell
python train.py --target_dataset 300W --model_type 68_points --train_file_path ./datasets/300W/train_data/list.txt
```

output:

```text
Epoch:[40/40], step:[127/129], loss:[0.709/0.732], time:418.097ms, lr:0.00001
Epoch:[40/40], step:[128/129], loss:[0.739/0.739], time:417.928ms, lr:0.00001
......
Epoch:[40/40], step:[129/129], loss:[0.729/0.734], time:418.322ms, lr:0.00001
```

### Evaluate Model

After training, you can use test set to evaluate the performance of your model.

Run eval.py to achieve this. The usage of model_type parameter is same as training process.

```text
python eval.py --model_type 68_points
```

output:

```text
nme:0.0641
ion:0.5747
ipn:0.3195
inference_cost_time: 0.014595
```

### Infer

At last, you can use your own image to test your model. Put your image in the infer_image folder, then run infer.py to
do inference.

```shell
python infer.py --model_type 68_points
```

**Result**

[外链图片转存失败,源站可能有防盗链机制,建议将图片保存下来直接上传(img-Hij0S1Lh-1658887682024)(./images/result.png)]
