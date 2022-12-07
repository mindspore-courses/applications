# ShuffleNet
ShuffleNet-v1 based on MindSpore

有关模型的介绍请参见"ShuffleNet-v1.ipynb"（训练过程代码请以以下脚本为准）.

训练脚本：
```python
预训练脚本：
python ./train.py --train_dataset_path ./data/cifar10
预训练评估脚本：
python ./eval.py --eval_dataset_path ./data/cifar10 --ckpt_path [ckpt]
迁移训练脚本：
python ./train.py --config_path=./transfer_config.yaml --resume=[ckpt] --dataset_path=./data/flower_photos
迁移训练评估脚本：
python ./eval.py --config_path=./transfer_config.yaml --dataset_path=./data/flower_photos
```

在ckpt文件夹下保存了预训练完成的模型shufflenetv1-250_195.ckpt和迁移训练完成的模型shufflenetv1_transfer_best.ckpt
