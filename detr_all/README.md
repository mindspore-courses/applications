# DETR(Detection Transformer)

 [DETR](https://link.zhihu.com/?target=https%3A//www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123460205.pdf) is an end-to-end target detection network based on Transformer proposed by Facebook and published in ECCV2020. Different from the popular faster RCNN and YOLO series of target detection models, DETR is an end-to-end target detection model, which does not need traditional manual design, such as anchor point generation, maximum suppression and other operations. DETR uses the transformer architecture and the newly designed loss function bipartite matching loss to directly reason the whole picture and output the target and category at the same time.

 DETR model is mainly composed of three parts: backbone network, transformer structure and FFN forward feedback network.

DETR adopts ResNet as the backbone network of feature extraction. The traditional convolution network or fully connected network has more or less the problem of information loss, which will also cause the gradient to disappear or explode, leading to the failure of deep network training. ResNet has solved this problem to some extent.

Transformer has been widely used since it was put forward in 2017. It has basically become a unified paradigm not only in the NLP field, but also in some visual fields, such as image classification, target detection, behavior recognition, etc., replacing CNN in some functions. As the pioneering work of Transformer used in the field of target detection, DETR uses the attention mechanism in transformer to obtain the global information of the image, which simplifies pipline of target detection.

FFN feedforward network is mainly composed of linear layers. Output a series of target locations and categories.

In order to make a series of goals output by the model correspond to the goals of ground truth to achieve the purpose of calculating loss, this paper adopts the classical bipartite matching algorithm-Hungarian algorithm. Hungarian algorithm can find the bipartite matching scheme that minimizes the total cost.

The overall process of the model: the image is input to ResNet to obtain the feature map, the feature map is converted to one dimension and the positional encoding is added, and then entered into the Transformer, and after encoder and decoder, the forward feedback network of the FFN is entered to obtain the probability distribution of some column positions and categories.

![network.png](./md_imgs/network2.png)

## Pretrained model

COCO val5k evaluation results and models：

|      | name     | backbone | inf_time | box AP | url                                                          | size |
| ---- | -------- | -------- | -------- | ------ | ------------------------------------------------------------ | ---- |
| 0    | DETR     | R50      | 0.223    | 42.1   | [ckpt](https://download.mindspore.cn/vision/detr/resume/resnet50.ckpt) | 159M |
| 1    | DETR-DC5 | R50      | 0.226    | 43.2   | [ckpt](https://download.mindspore.cn/vision/detr/resume/resnet50_dc.ckpt) | 159M |
| 2    | DETR     | R101     | 0.255    | 43.6   | [ckpt](https://download.mindspore.cn/vision/detr/resume/resnet101.ckpt) | 232M |
| 3    | DETR-DC5 | R101     | 0.259    | 44.9   | [ckpt](https://download.mindspore.cn/vision/detr/resume/resnet101_dc.ckpt) | 232M |

COCO panoptic val5k evaluation results and models:

|      | name     | backbone | box AP | segm AP | PQ   | url                                                          | size  |
| ---- | -------- | -------- | ------ | ------- | ---- | ------------------------------------------------------------ | ----- |
| 0    | DETR     | R50      | 38.8   | 32.5    | 43.6 | [ckpt](https://download.mindspore.cn/vision/detr/resume/resnet50_seg.ckpt) | 165M |
| 1    | DETR-DC5 | R50      | 40.1   | 33.4    | 44.7 | [ckpt](https://download.mindspore.cn/vision/detr/resume/resnet50_dc_seg.ckpt) | 165M |
| 2    | DETR     | R101     | 40.1   | 34.4    | 45.1 | [ckpt](https://download.mindspore.cn/vision/detr/resume/resnet101_seg.ckpt) | 237M |

## Training Parameter description

| Parameter         | Description                           | Default              |
| --------------- | ------------------------------ | ----------------- |
| --device        | Device type['CPU','GPU','Ascend'] | 'CPU'             |
| lr              | Base learning rate             | 1e-4              |
| weight_decay    | Control weight decay speed                 | 1e-6              |
| epoch           | Number of epoch                       | 100               |
| batch           | Number of batch size                       | 2                 |
| coco_dir        | Path of coco                 | './coco'          |
| pano_dir        | Path of coco_panoptic                 | './coco_panoptic' |
| resnet          | resnet type                     | resnet50          |
| dilation        | Dilated Convolution               | False             |
| is_segmentation | segmentation                   | False             |
| checkpoint_path | Path of Pre-training model                   | './checkpoint'    |

## Dataset

This example uses the COCO dataset as the training set and validation set. Go to the official url: <http://mscoco.org/> Download the following 4 files, the corresponding file size and its corresponding link are as follows:

train2017 images: (18GB) <http://images.cocodataset.org/zips/train2017.zip>

val2017 images: (1GB) <http://images.cocodataset.org/zips/val2017.zip>

train2017/val2017 annotations: (241MB) <http://images.cocodataset.org/annotations/annotations_trainval2017.zip>

Panoptic train2017/val2017 annotations: (821MB) <http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip>

The training set 118287 images and the validation set 5,000 pictures. The dataset image is shown in the following figure:

Please put the extracted dataset under ./data/, the file directory is as follows:

```shell
data/coco/
    annotations/  # annotation json files
    train2017/    # train images
    val2017/      # val images

data/coco_panoptic/
    annotations/  # annotation json files
    panoptic_train2017/    # train panoptic annotations
    panoptic_val2017/      # val panoptic annotations
```

## Train Model

```shell
python train.py --coco_dir ./coco --checkpoint_path ./checkpoint
```

```shell
loading annotations into memory...
Done (t=25.12s)
creating index...
index created!
epoch: 1 step: 1, loss is 11.121121810283512
epoch: 1 step: 2, loss is 12.20494846560061
epoch: 1 step: 3, loss is 12.587393889735853
epoch: 1 step: 4, loss is 11.599971771240234
epoch: 1 step: 5, loss is 9.60177993774414
......
epoch: 1 step: 936, loss is 4.0714711009391715
epoch: 1 step: 937, loss is 4.941065043210983
epoch: 1 step: 938, loss is 6.586935043334961
epoch: 1 step: 939, loss is 5.4569307619240135
epoch: 1 step: 940, loss is 7.609074387059081
```

## Evaluate Model

```shell
python src/eval.py --coco_dir /data0/my_coco --result_dir ./result
```

```shell
Evaluate annotation type *bbox*
DONE (t=56.15s).
Accumulating evaluation results...
DONE (t=10.70s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.421
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.623
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.444
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.459
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.334
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.532
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.574
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.318
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.629
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.805
```

## Infer

```shell
python infer.py --img_path ../images/000000056288.jpg --resume_path ../resume/resnet50.ckpt
```

## Result

![detr.png](./src/img_detr.png)
