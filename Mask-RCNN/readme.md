# 模型介绍
Mask R-CNN通过添加一个与现有目标检测框回归并行的，用于预测目标掩码的分支来扩展Faster R-CNN，通过添加一个用于在每个感兴趣区域（RoI）上预测分割掩码的分支来扩展Faster R-CNN，就是在每个感兴趣区域（RoI）进行一个二分类的语义分割，在这个感兴趣区域同时做目标检测和分割，这个分支与用于分类和目标检测框回归的分支并行执行，如下图所示（用于目标分割的Mask R-CNN框架）：
![model](/img/model.jpg)
掩码分支是作用于每个RoI的小FCN，以像素到像素的方式预测分割掩码，可是要在ROI区域进行一个mask分割，存在一个问题，Faster R-CNN不是为网络输入和输出之间的像素到像素对齐而设计的，如果直接拿Faster R-CNN得到的ROI进行mask分割，那么像素到像素的分割可能不精确，因为应用到目标检测上的核心操作执行的是粗略的空间量化特征提取，直接分割出来的mask存在错位的情况，所以作者提出了简单的，量化无关的层，称为RoIAlign(ROI对齐)，可以保留精确的空间位置，可以将掩码(mask)准确度提高10％至50％。
[论文:](http://cn.arxiv.org/pdf/1703.06870v3)"MaskRCNN"

## 模型结构
Mask R-CNN由两个阶段组成，称为区域提议网络（RPN）的第一阶段提出候选目标边界框。第二阶段本质上就是Fast R-CNN，它使用来自候选框架中的RoIPool来提取特征并进行分类和边界框回归，Mask R-CNN还为每个RoI输出二进制掩码。
![framework](/img/framework.jpg)
模型整体结构组成为：骨干网络，特征金字塔网络(FPN)、区域提议网络以及Fast R-CNN。骨干网络和特征金字塔网络提取出图像的特征，然后区域提议网络在特征图上筛选出候选框，结合特征图和候选框后进行RoI Align，此后送入三个分支网络，分别获取分类结果、目标框以及目标区域的mask。

下图是以ResNet50为骨干网络加上FPN的网络结构：
![](/img/resnet%2Bfpn.jpg)

此处推荐三个视频用于读者加深理解：[FPN](https://www.bilibili.com/video/BV1dh411U7D9?vd_source=2277cd0edf083354294094185f8857ab)、[Faster RCNN](https://www.bilibili.com/video/BV1af4y1m7iL?p=3&vd_source=2277cd0edf083354294094185f8857ab)、[Mask RCNN](https://www.bilibili.com/video/BV1ZY411774T?vd_source=2277cd0edf083354294094185f8857ab)。

## 模型优点
Mask R-CNN训练简单，只增加了Faster R-CNN少量的开销即可处理语义分割问题，运行速度为5fps。此外Mask R-CNN易于推广到其它任务中。

# 案例实现
## 环境准备与数据读取
本案例基于MindSpore-GPU版本实现，在GPU上完成模型训练。

案例实现所使用的数据为coco数据集，可从https://cocodataset.org/#download下载，需要下载对应的Image和Annotations。下载下来的Annotations的结构如图:
![](/img/struct.jpg)
若需要调整训练所用的数据量，只需要调整image的选项即可。

**注意事项**：本案例需要安装pycocotools，若在windows上，使用``pip install pycocotools``安装会提示缺少c++库，可以使用``pip install pycocotools-windows``安装。在linux下安装时，可能会报缺少Cython的错误，先使用``pip install cython``安装好Cython即可。

## 数据集创建
接来下是创建数据集部分，首先需要将Annotations中的instances_(train/val)2017.json文件读入内存，然后根据images中的图片信息读取图片并解析mask信息然后将这些信息存入mindrecord文件(用于加速数据读取)。最后根据设定的batch size等信息创建MindDataset并组织数据成需要的形式，此过程会返回一个指向数据集的迭代器。至此完成数据集的创建，上述过程的代码对应如下：
```
def data_to_mindrecord_byte_image(dataset="coco", is_training=True, prefix="maskrcnn.mindrecord", file_num=8):
    mindrecord_dir = config.mindrecord_dir
    mindrecord_path = os.path.join(mindrecord_dir, prefix)

    writer = FileWriter(mindrecord_path, file_num)
    if dataset == "coco":
        image_files, image_anno_dict, masks, masks_shape = create_coco_label(is_training)
    else:
        print("Error unsupported other dataset")
        return

    maskrcnn_json = {
        "image": {"type": "bytes"},
        "annotation": {"type": "int32", "shape": [-1, 6]},
        "mask": {"type": "bytes"},
        "mask_shape": {"type": "int32", "shape": [-1]},
    }
    writer.add_schema(maskrcnn_json, "maskrcnn_json")

    image_files_num = len(image_files)
    for ind, image_name in enumerate(image_files):
        with open(image_name, 'rb') as f:
            img = f.read()
        annos = np.array(image_anno_dict[image_name], dtype=np.int32)
        mask = masks[image_name]
        mask_shape = masks_shape[image_name]
        row = {"image": img, "annotation": annos, "mask": mask, "mask_shape": mask_shape}
        if (ind + 1) % 10 == 0:
            print("writing {}/{} into mindrecord".format(ind + 1, image_files_num))
        writer.write_raw_data([row])
    writer.commit()

def create_coco_label(is_training):
    from pycocotools.coco import COCO

    coco_root = config.coco_root
    data_type = config.val_data_type
    if is_training:
        data_type = config.train_data_type

    # Classes need to train or test.
    train_cls = config.coco_classes

    train_cls_dict = {}
    for i, cls in enumerate(train_cls):
        train_cls_dict[cls] = i

    anno_json = os.path.join(coco_root, config.instance_set.format(data_type))

    coco = COCO(anno_json)
    classs_dict = {}
    cat_ids = coco.loadCats(coco.getCatIds())
    for cat in cat_ids:
        classs_dict[cat["id"]] = cat["name"]

    image_ids = coco.getImgIds()
    image_files = []
    image_anno_dict = {}
    masks = {}
    masks_shape = {}
    images_num = len(image_ids)
    for ind, img_id in enumerate(image_ids):
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        image_path = os.path.join(coco_root, data_type, file_name)
        if not os.path.isfile(image_path):
            print("{}/{}: {} is in annotations but not exist".format(ind + 1, images_num, image_path))
            continue
        anno_ids = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = coco.loadAnns(anno_ids)
        image_path = os.path.join(coco_root, data_type, file_name)
        annos = []
        instance_masks = []
        image_height = coco.imgs[img_id]["height"]
        image_width = coco.imgs[img_id]["width"]
        if (ind + 1) % 10 == 0:
            print("{}/{}: parsing annotation for image={}".format(ind + 1, images_num, file_name))
        if not is_training:
            image_files.append(image_path)
            image_anno_dict[image_path] = np.array([0, 0, 0, 0, 0, 1])
            masks[image_path] = np.zeros([1, 1, 1], dtype=bool).tobytes()
            masks_shape[image_path] = np.array([1, 1, 1], dtype=np.int32)
        else:
            for label in anno:
                bbox = label["bbox"]
                class_name = classs_dict[label["category_id"]]
                if class_name in train_cls:
                    # get coco mask
                    m = annToMask(label, image_height, image_width)
                    if m.max() < 1:
                        print("all black mask!!!!")
                        continue
                    # Resize mask for the crowd
                    if label['iscrowd'] and (m.shape[0] != image_height or m.shape[1] != image_width):
                        m = np.ones([image_height, image_width], dtype=bool)
                    instance_masks.append(m)

                    # get coco bbox
                    x1, x2 = bbox[0], bbox[0] + bbox[2]
                    y1, y2 = bbox[1], bbox[1] + bbox[3]
                    annos.append([x1, y1, x2, y2] + [train_cls_dict[class_name]] + [int(label["iscrowd"])])
                else:
                    print("not in classes: ", class_name)

            image_files.append(image_path)
            if annos:
                image_anno_dict[image_path] = np.array(annos)
                instance_masks = np.stack(instance_masks, axis=0).astype(bool)
                masks[image_path] = np.array(instance_masks).tobytes()
                masks_shape[image_path] = np.array(instance_masks.shape, dtype=np.int32)
            else:
                print("no annotations for image ", file_name)
                image_anno_dict[image_path] = np.array([0, 0, 0, 0, 0, 1])
                masks[image_path] = np.zeros([1, image_height, image_width], dtype=bool).tobytes()
                masks_shape[image_path] = np.array([1, image_height, image_width], dtype=np.int32)

    return image_files, image_anno_dict, masks, masks_shape

def create_maskrcnn_dataset(mindrecord_file, batch_size=2, device_num=1, rank_id=0,
                            is_training=True, num_parallel_workers=2):
    cv2.setNumThreads(0)
    de.config.set_prefetch_size(8)
    ds = de.MindDataset(mindrecord_file, columns_list=["image", "annotation", "mask", "mask_shape"],
                        num_shards=device_num, shard_id=rank_id,
                        num_parallel_workers=4, shuffle=is_training)

    decode = vision.Decode()
    ds = ds.map(operations=decode, input_columns=["image"])
    compose_map_func = (lambda image, annotation, mask, mask_shape:
                        preprocess_fn(image, annotation, mask, mask_shape, is_training))

    if is_training:
        ds = ds.map(operations=compose_map_func,
                    input_columns=["image", "annotation", "mask", "mask_shape"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    column_order=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    python_multiprocessing=False,
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True, pad_info={"mask": ([config.max_instance_count, None, None], 0)})

    else:
        ds = ds.map(operations=compose_map_func,
                    input_columns=["image", "annotation", "mask", "mask_shape"],
                    output_columns=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    column_order=["image", "image_shape", "box", "label", "valid_num", "mask"],
                    num_parallel_workers=num_parallel_workers)
        ds = ds.batch(batch_size, drop_remainder=True)

    return ds
```

## 模型构建
本案例实现的Mask R-CNN结构和论文中的结构大体相同，骨干网络采用的是ResNet50。具体的网络结构，读者根据模型结构介绍中的图片以及论文原文中提供的mask分支结构可以有个清晰的认知，此处就不放具体的网络结构图了。

在具体的实现上，MindSpore和pytorch差异不大，只是一些方法名上的替换，具体的API映射可见https://www.mindspore.cn/docs/zh-CN/r1.8/note/api_mapping/pytorch_api_mapping.html#torch-nn。
```
class MaskRCNN(nn.Cell):
    def __init__(self, config):
        super(MaskRCNN, self).__init__()

        if context.get_context("device_target") == "Ascend":
            self.cast_type = mstype.float16
            self.np_cast_type = np.float16
        else:
            self.cast_type = mstype.float32
            self.np_cast_type = np.float32

        self.train_batch_size = config.batch_size
        self.num_classes = config.num_classes
        self.anchor_scales = config.anchor_scales
        self.anchor_ratios = config.anchor_ratios
        self.anchor_strides = config.anchor_strides
        self.target_means = tuple(config.rcnn_target_means)
        self.target_stds = tuple(config.rcnn_target_stds)

        # Anchor generator
        anchor_base_sizes = None
        self.anchor_base_sizes = list(
            self.anchor_strides) if anchor_base_sizes is None else anchor_base_sizes

        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGenerator(anchor_base, self.anchor_scales, self.anchor_ratios))

        self.num_anchors = len(self.anchor_ratios) * len(self.anchor_scales)

        featmap_sizes = config.feature_shapes
        assert len(featmap_sizes) == len(self.anchor_generators)

        self.anchor_list = self.get_anchors(featmap_sizes)

        # Backbone resnet50
        self.backbone = ResNet(ResidualBlock,
                                  config.resnet_block,
                                  config.resnet_in_channels,
                                  config.resnet_out_channels,
                                  False)

        # Fpn
        self.fpn_ncek = FeatPyramidNeck(config.fpn_in_channels,
                                        config.fpn_out_channels,
                                        config.fpn_num_outs,
                                        config.feature_shapes)

        # Rpn and rpn loss
        self.gt_labels_stage1 = Tensor(np.ones((self.train_batch_size, config.num_gts)).astype(np.uint8))
        self.rpn_with_loss = RPN(config,
                                 self.train_batch_size,
                                 config.rpn_in_channels,
                                 config.rpn_feat_channels,
                                 config.num_anchors,
                                 config.rpn_cls_out_channels)

        # Proposal
        self.proposal_generator = Proposal(config,
                                           self.train_batch_size,
                                           config.activate_num_classes,
                                           config.use_sigmoid_cls)
        self.proposal_generator.set_train_local(config, True)
        self.proposal_generator_test = Proposal(config,
                                                config.test_batch_size,
                                                config.activate_num_classes,
                                                config.use_sigmoid_cls)
        self.proposal_generator_test.set_train_local(config, False)

        # Assign and sampler stage two
        self.bbox_assigner_sampler_for_rcnn = BboxAssignSampleForRcnn(config, self.train_batch_size,
                                                                      config.num_bboxes_stage2, True)
        self.decode = P.BoundingBoxDecode(max_shape=(768, 1280), means=self.target_means, \
                                          stds=self.target_stds)

        # Roi
        self.init_roi(config)

        # Rcnn
        self.rcnn_cls = RcnnCls(config, self.train_batch_size, self.num_classes)
        self.rcnn_mask = RcnnMask(config, self.train_batch_size, self.num_classes)

        # Op declare
        self.squeeze = P.Squeeze()
        self.cast = P.Cast()

        self.concat = P.Concat(axis=0)
        self.concat_1 = P.Concat(axis=1)
        self.concat_2 = P.Concat(axis=2)
        self.reshape = P.Reshape()
        self.select = P.Select()
        self.greater = P.Greater()
        self.transpose = P.Transpose()

        # Test mode
        self.init_test_mode(config)

        # Improve speed
        self.concat_start = min(self.num_classes - 2, 55)
        self.concat_end = (self.num_classes - 1)

        # Init tensor
        self.init_tensor(config)
    
        def construct(self, img_data, img_metas, gt_bboxes, gt_labels, gt_valids, gt_masks):
        """Construct for Mask R-CNN net."""
        x = self.backbone(img_data)
        x = self.fpn_ncek(x)

        rpn_loss, cls_score, bbox_pred, rpn_cls_loss, rpn_reg_loss, _ = self.rpn_with_loss(x,
                                                                                           img_metas,
                                                                                           self.anchor_list,
                                                                                           gt_bboxes,
                                                                                           self.gt_labels_stage1,
                                                                                           gt_valids)

        if self.training:
            proposal, proposal_mask = self.proposal_generator(cls_score, bbox_pred, self.anchor_list)
        else:
            proposal, proposal_mask = self.proposal_generator_test(cls_score, bbox_pred, self.anchor_list)

        gt_labels = self.cast(gt_labels, mstype.int32)
        gt_valids = self.cast(gt_valids, mstype.int32)
        bboxes_tuple = ()
        deltas_tuple = ()
        labels_tuple = ()
        mask_tuple = ()

        pos_bboxes_tuple = ()
        pos_mask_fb_tuple = ()
        pos_labels_tuple = ()
        pos_mask_tuple = ()

        if self.training:
            for i in range(self.train_batch_size):
                gt_bboxes_i = self.squeeze(gt_bboxes[i:i + 1:1, ::])

                gt_labels_i = self.squeeze(gt_labels[i:i + 1:1, ::])
                gt_labels_i = self.cast(gt_labels_i, mstype.uint8)

                gt_valids_i = self.squeeze(gt_valids[i:i + 1:1, ::])
                gt_valids_i = self.cast(gt_valids_i, mstype.bool_)

                gt_masks_i = self.squeeze(gt_masks[i:i + 1:1, ::])
                gt_masks_i = self.cast(gt_masks_i, mstype.bool_)

                bboxes, deltas, labels, mask, pos_bboxes, pos_mask_fb, pos_labels, pos_mask = \
                    self.bbox_assigner_sampler_for_rcnn(gt_bboxes_i,
                                                        gt_labels_i,
                                                        proposal_mask[i],
                                                        proposal[i][::, 0:4:1],
                                                        gt_valids_i,
                                                        gt_masks_i)
                bboxes_tuple += (bboxes,)
                deltas_tuple += (deltas,)
                labels_tuple += (labels,)
                mask_tuple += (mask,)

                pos_bboxes_tuple += (pos_bboxes,)
                pos_mask_fb_tuple += (pos_mask_fb,)
                pos_labels_tuple += (pos_labels,)
                pos_mask_tuple += (pos_mask,)

            bbox_targets = self.concat(deltas_tuple)
            rcnn_labels = self.concat(labels_tuple)
            bbox_targets = F.stop_gradient(bbox_targets)
            rcnn_labels = F.stop_gradient(rcnn_labels)
            rcnn_labels = self.cast(rcnn_labels, mstype.int32)

            rcnn_pos_masks_fb = self.concat(pos_mask_fb_tuple)
            rcnn_pos_masks_fb = F.stop_gradient(rcnn_pos_masks_fb)
            rcnn_pos_labels = self.concat(pos_labels_tuple)
            rcnn_pos_labels = F.stop_gradient(rcnn_pos_labels)
            rcnn_pos_labels = self.cast(rcnn_pos_labels, mstype.int32)
        else:
            mask_tuple += proposal_mask
            bbox_targets = proposal_mask
            rcnn_labels = proposal_mask

            rcnn_pos_masks_fb = proposal_mask
            rcnn_pos_labels = proposal_mask
            for p_i in proposal:
                bboxes_tuple += (p_i[::, 0:4:1],)

        bboxes_all, rois, pos_rois = self.rois(bboxes_tuple, pos_bboxes_tuple)

        if self.training:
            roi_feats = self.roi_align(rois,
                                       self.cast(x[0], mstype.float32),
                                       self.cast(x[1], mstype.float32),
                                       self.cast(x[2], mstype.float32),
                                       self.cast(x[3], mstype.float32))
        else:
            roi_feats = self.roi_align_test(rois,
                                            self.cast(x[0], mstype.float32),
                                            self.cast(x[1], mstype.float32),
                                            self.cast(x[2], mstype.float32),
                                            self.cast(x[3], mstype.float32))


        roi_feats = self.cast(roi_feats, self.cast_type)
        rcnn_masks = self.concat(mask_tuple)
        rcnn_masks = F.stop_gradient(rcnn_masks)
        rcnn_mask_squeeze = self.squeeze(self.cast(rcnn_masks, mstype.bool_))

        rcnn_pos_masks = self.concat(pos_mask_tuple)
        rcnn_pos_masks = F.stop_gradient(rcnn_pos_masks)
        rcnn_pos_mask_squeeze = self.squeeze(self.cast(rcnn_pos_masks, mstype.bool_))

        rcnn_cls_loss, rcnn_reg_loss = self.rcnn_cls(roi_feats,
                                                     bbox_targets,
                                                     rcnn_labels,
                                                     rcnn_mask_squeeze)

        if self.training:
            return self.get_output_train(pos_rois, x, rcnn_pos_labels, rcnn_pos_mask_squeeze, rcnn_pos_masks_fb,
                                         rpn_loss, rpn_cls_loss, rpn_reg_loss, rcnn_cls_loss, rcnn_reg_loss)

        return self.get_output_eval(x, bboxes_all, rcnn_cls_loss, rcnn_reg_loss, rcnn_masks, img_metas)
```

## 评估指标
本案例的评估指标使用的pycocotools工具包中实现的指标，使用了mAP和AR。
- mAP:AP的计算仅涉及一类。但是，在物体检测中，通常有k个类。平均平均精度（mAP）定义为k类AP的平均值:
$$
mAP = \frac{ {\textstyle \sum_{i=1}^{K}}AP_i}{K}
$$
- 像AP一样，平均召回率（AR）也是可用于比较检测器性能的数值指标。本质上，AR是可以计算为召回-IoU曲线下面积的两倍：
$$
AR = 2\int_{0.5}^{1} recall(o)do
$$
应该注意的是，出于其最初的目的（Hosang等人，2016年），召回率-IoU曲线无法区分不同的类别。但是，COCO挑战做出了这样的区分，并且其AR指标是按类别计算的，就像AP一样。
```
def coco_eval(result_files, result_types, coco, max_dets=(100, 300, 1000), single_result=False):
    anns = json.load(open(result_files['bbox']))
    if not anns:
        return summary_init
    if isinstance(coco, str):
        coco = COCO(coco)
    assert isinstance(coco, COCO)

    for res_type in result_types:
        result_file = result_files[res_type]
        assert result_file.endswith('.json')

        coco_dets = coco.loadRes(result_file)
        gt_img_ids = coco.getImgIds()
        det_img_ids = coco_dets.getImgIds()
        iou_type = 'bbox' if res_type == 'proposal' else res_type
        cocoEval = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)

        tgt_ids = gt_img_ids if not single_result else det_img_ids

        if single_result:
            res_dict = dict()
            for id_i in tgt_ids:
                cocoEval = COCOeval(coco, coco_dets, iou_type)
                if res_type == 'proposal':
                    cocoEval.params.useCats = 0
                    cocoEval.params.maxDets = list(max_dets)

                cocoEval.params.imgIds = [id_i]
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                res_dict.update({coco.imgs[id_i]['file_name']: cocoEval.stats[1]})

        cocoEval = COCOeval(coco, coco_dets, iou_type)
        if res_type == 'proposal':
            cocoEval.params.useCats = 0
            cocoEval.params.maxDets = list(max_dets)

        cocoEval.params.imgIds = tgt_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        summary_metrics = {
            'Precision/mAP': cocoEval.stats[0],
            'Precision/mAP@.50IOU': cocoEval.stats[1],
            'Precision/mAP@.75IOU': cocoEval.stats[2],
            'Precision/mAP (small)': cocoEval.stats[3],
            'Precision/mAP (medium)': cocoEval.stats[4],
            'Precision/mAP (large)': cocoEval.stats[5],
            'Recall/AR@1': cocoEval.stats[6],
            'Recall/AR@10': cocoEval.stats[7],
            'Recall/AR@100': cocoEval.stats[8],
            'Recall/AR@100 (small)': cocoEval.stats[9],
            'Recall/AR@100 (medium)': cocoEval.stats[10],
            'Recall/AR@100 (large)': cocoEval.stats[11],
        }

    return summary_metrics
```

## 模型训练、评估及结果可视化
由于回调形式的模型训练不便于定位错误位置，因此本案例实现采用for循环的方式进行网络训练。for循环形式的训练整体上和pytorch差不多，但是需要注意的是mindspore将优化器进行了使用上的简化，不再需要像pytorch一样手动调用step，而是通过ops.depend将优化器的归零和更新进行了包装。

评估时，会对test_dir目录下的所有图片进行评估，若需要修改评估对象，可以修改config中的test_dir，同时也需要修改test_annotations中的标注文件。

结果可视化需要在评估之后才能调用，该步骤依赖于评估生成的结果文件。
```
def train():
    rank = 0
    prefix = "MaskRcnn.mindrecord"
    mindrecord_dir = 'val2017'
    mindrecord_file = os.path.join(mindrecord_dir, prefix + "0")
    if rank == 0 and not os.path.exists(mindrecord_file):
        create_mindrecord_dir(prefix, mindrecord_dir, mindrecord_file)
    dataset = create_maskrcnn_dataset(mindrecord_file, batch_size=config.batch_size, device_num=1, rank_id=0)
    net = MaskRCNN(config)
    net = net.set_train()
    loss = LossNet()
    lr = Tensor(0.0001, mstype.float32)
    opt = Momentum(params=net.trainable_params(), learning_rate=lr, momentum=0.9)

    def forward_fn(img_data, img_metas, gt_bboxes, gt_labels, gt_num, gt_mask):
        output = net(img_data, img_metas, gt_bboxes, gt_labels, gt_num, gt_mask)
        l = loss(*output)
        return l
    grad_fn = ops.value_and_grad(forward_fn, None, opt.parameters, has_aux=False)

    def train_step(img_data, img_metas, gt_bboxes, gt_labels, gt_num, gt_mask):
        (loss), grads = grad_fn(img_data, img_metas, gt_bboxes, gt_labels, gt_num, gt_mask)
        loss = ops.depend(loss, opt(grads))
        return loss
    for epoch in range(config.epoch_size):
        step = 0
        for data in dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
            img_data = data['image']
            img_metas = data['image_shape']
            gt_bboxes = data['box']
            gt_labels = data['label']
            gt_num = data['valid_num']
            gt_mask = data["mask"]
            l = train_step(Tensor(img_data, dtype=mstype.float32), Tensor(img_metas, dtype=mstype.float32),
                              Tensor(gt_bboxes, dtype=mstype.float32), Tensor(gt_labels, dtype=mstype.float32),
                              Tensor(gt_num, dtype=mstype.float32), Tensor(gt_mask, dtype=mstype.float32))
            print("epoch:", epoch, " step:", step, " loss:", l)
            step += 1
    ms.save_checkpoint(net, "./ckpt_" + str(rank) + "/mask_rcnn.ckpt")
    print('---------train done-----------')


def eval_():
    device_target = config.device_target
    context.set_context(mode=context.GRAPH_MODE, device_target=device_target)

    config.mindrecord_dir = os.path.join(config.coco_root, config.test_dir)
    prefix = "MaskRcnn_eval.mindrecord"
    mindrecord_dir = config.mindrecord_dir
    mindrecord_file = os.path.join(mindrecord_dir, prefix)
    if not os.path.exists(mindrecord_file):
        if not os.path.isdir(mindrecord_dir):
            os.makedirs(mindrecord_dir)
        if os.path.isdir(config.coco_root):
            print("Create Mindrecord.")
            data_to_mindrecord_byte_image("coco", False, prefix, file_num=1)
            print("Create Mindrecord Done, at {}".format(mindrecord_dir))
        else:
            print("coco_root not exits.")

    print("Start Eval!")

    ds = create_maskrcnn_dataset(mindrecord_file, batch_size=config.test_batch_size, is_training=False)

    net = MaskRCNN(config)
    param_dict = load_checkpoint('./ckpt_0/mask_rcnn.ckpt')
    load_param_into_net(net, param_dict)
    net.set_train(False)

    eval_iter = 0
    total = ds.get_dataset_size()
    outputs = []
    dataset_coco = COCO('test_annotations/instances_val2017.json')

    print("\n========================================\n")
    print("total images num: ", total)
    print("Processing, please wait a moment.")
    max_num = 128
    for data in ds.create_dict_iterator(output_numpy=True, num_epochs=1):

        img_data = data['image']
        img_metas = data['image_shape']
        gt_bboxes = data['box']
        gt_labels = data['label']
        gt_num = data['valid_num']
        gt_mask = data["mask"]

        start = time.time()

        # run net
        output = net(Tensor(img_data, dtype=mstype.float32), Tensor(img_metas, dtype=mstype.float32), Tensor(gt_bboxes, dtype=mstype.float32),
                     Tensor(gt_labels, dtype=mstype.float32), Tensor(gt_num, dtype=mstype.float32), Tensor(gt_mask, dtype=mstype.float32))
        end = time.time()
        print("Iter {} cost time {}".format(eval_iter, end - start))

        # output
        all_bbox = output[0]
        all_label = output[1]
        all_mask = output[2]
        all_mask_fb = output[3]
        print(all_bbox.shape)
        print(all_mask.shape, np.sum(all_mask.asnumpy()[0]))

        for j in range(config.test_batch_size):
            all_bbox_squee = np.squeeze(all_bbox.asnumpy()[j, :, :])
            all_label_squee = np.squeeze(all_label.asnumpy()[j, :, :])
            all_mask_squee = np.squeeze(all_mask.asnumpy()[j, :, :])
            all_mask_fb_squee = np.squeeze(all_mask_fb.asnumpy()[j, :, :, :])

            all_bboxes_tmp_mask = all_bbox_squee[all_mask_squee, :]
            all_labels_tmp_mask = all_label_squee[all_mask_squee]
            all_mask_fb_tmp_mask = all_mask_fb_squee[all_mask_squee, :, :]

            if all_bboxes_tmp_mask.shape[0] > max_num:
                inds = np.argsort(-all_bboxes_tmp_mask[:, -1])
                inds = inds[:max_num]
                all_bboxes_tmp_mask = all_bboxes_tmp_mask[inds]
                all_labels_tmp_mask = all_labels_tmp_mask[inds]
                all_mask_fb_tmp_mask = all_mask_fb_tmp_mask[inds]

            bbox_results = bbox2result_1image(all_bboxes_tmp_mask, all_labels_tmp_mask, config.num_classes)
            segm_results = get_seg_masks(all_mask_fb_tmp_mask, all_bboxes_tmp_mask, all_labels_tmp_mask, img_metas[j],
                                         True, config.num_classes)
            outputs.append((bbox_results, segm_results))

            eval_iter = eval_iter + 1

    eval_types = ["bbox", "segm"]
    result_files = results2json(dataset_coco, outputs, "./results.pkl")
    metrics = coco_eval(result_files, eval_types, dataset_coco, single_result=False)
    print(metrics)


def get_eval_result(bbox_file, segm_file, ann_file, img_name, img_path):
    """ Get metrics result according to the annotation file and result file"""
    with open(bbox_file) as b, open(segm_file) as s:
        bboxes = json.load(b)
        segms = json.load(s)
        data_coco = COCO(ann_file)
        img_id = -1
        for k, v in data_coco.imgs.items():
            if v['file_name'] == img_name:
                img_id = k
        img = cv2.imread(img_path + "/" + img_name)
        img1 = img.copy()
        for d in bboxes:
            if d['image_id'] == img_id:
                box = d['bbox']
                x, y, w, h = box
                a = (int(x), int(y))
                b = (int(x + w), int(y + h))
                img1 = cv2.rectangle(img1, a, b, (0, 255, 255), 2)
                img1 = cv2.putText(img1, "{} {:.3f}".format(config.coco_classes[int(d['category_id'])], d['score']),
                                   (b[0], a[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.imshow("detect", img1)
        cv2.waitKey()
        color = (0, 0.6, 0.6)
        for d in segms:
            if d['image_id'] == img_id:
                mask = maskUtils.decode(d['segmentation'])
                mask = np.where(mask > 0, 1, 0).astype(np.uint8)
                for c in range(3):
                    img[:, :, c] = np.where(mask == 1, img[:, :, c] * 0.5 + 0.5 * color[c] * 255, img[:, :, c])
        cv2.imshow("mask", img)
        cv2.waitKey()
```

## 本案例脚本的使用方式
通过将"train"/"eval"/"infer"赋给mode即可运行训练/评估/推理。eval会对test_img中的所有图片进行指标评估，然后将标注框和分割矩阵存放到根目录下的两个json文件中。infer则会根据eval的结果，对指定图片的检测和分割进行可视化展示。

所有的配置参数均在config类中，可以根据需要进行修改。


# 总结
本案例基于MindSpore框架针对coco数据集，完成了数据读取、数据集创建、Mask R-CNN模型构建，进行了模型训练和评估，顺利完成了预测结果的输出。通过此案例进一步加深了对Mask R-CNN模型结构和特性的理解，并结合MindSpore框架提供的文档和教程，掌握了利用Mindspore框架实现特定案例的流程，以及多种API的使用方法，为以后在实际场景中应用MindSpore框架提供支持。