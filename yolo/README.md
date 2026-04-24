# YOLO11 应用案例

本目录提供基于 MindSpore 适配实现的 `ultralytics` YOLO11 应用案例，包含以下四类任务：

- 图像分类
- 目标检测
- 实例分割
- 姿态估计

每个案例均提供训练、评估与推理的完整流程

## 环境准备

运行本目录下案例前，请先准备好以下环境：

- Python >= 3.9, < 3.12
- MindSpore >= 2.7.1
- CANN >= 8.1.RC1
- Ascend 环境可用
- 已安装基于 MindSpore 适配实现的 `ultralytics` 运行环境

其中，MindSpore 安装请参考官方文档：
https://www.mindspore.cn/install

说明：

1. 本案例调用的 `ultralytics` 为基于 MindSpore 适配实现的版本，请确保 `from ultralytics import YOLO` 可正常导入。
2. 当前 `applications` 仓主要提供应用案例文件，不单独包含 `ultralytics` 适配实现源码。
3. 运行本案例前，请先在对应环境中完成 `ultralytics` 适配实现及相关依赖的安装与配置，具体步骤可参考 MindNLP 仓库中 `src/mindnlp/ultralytics/readme.md` 的说明。
4. 若后续 `ultralytics` 能力正式并入 MindNLP，可根据实际集成方式调整安装与导入说明。

## 数据与权重准备

默认使用以下数据集配置文件：

- 检测：`cfg/datasets/coco128.yaml`
- 分类：`cfg/datasets/imagenette2-160.yaml`
- 分割：`cfg/datasets/coco128-seg.yaml`
- 姿态：`cfg/datasets/coco8-pose.yaml`

请根据对应 YAML 配置文件下载并放置数据集。推理阶段会优先使用数据集中的样例图片；如果当前环境下没有可用测试图片，案例代码会自动下载示例图片。



## Notebook 列表

- `yolo11_classification_train_eval_infer.ipynb`：YOLO11 目标检测案例
- `yolo11_detection_train_eval_infer.ipynb`：YOLO11 图像分类案例
- `yolo11_pose_train_eval_infer.ipynb`：YOLO11 实例分割案例
- `yolo11_segmentation_train_eval_infer.ipynb`：YOLO11 姿态估计案例
