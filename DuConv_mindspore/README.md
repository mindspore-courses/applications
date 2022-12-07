#
#《Proactive Human-Machine Conversation with Explicit Conversation Goals》论文：https://arxiv.org/abs/1906.05572v2
#

#
# jupyter文件夹内存放着项目ipynb格式的jupyter notebook格式，使用说明：
#

1、首先逐步运行DuConv_DataProcess.ipynb文件来进行数据预处理，从而将gz类型的数据集下载到本地并进行处理，最终生成mindrecord类型的训练集、验证集和测试集。

2、逐步运行DuConv_train_predict.ipynb文件来完成模型的训练和预测，训练完成的模型存在save_model目录下，预测出的结果以及评分存在output目录下。最终对预测的结果的各项指标进行评估，并将评估结果打印出来。



#
# 项目代码使用说明：
#
#### 第一部分：环境配置：安装GPU版本MindSpore（这里可以使用官网安装教程，注意python<3.9这里用的是python3.8，mindspore使用版本1.8.1）

```bash
pip install mindspore-gpu==1.8.1
```

#### 第二部分：下载和预处理数据
```bash
# download dataset
bash scripts/download_dataset.sh
# build dataset
bash scripts/build_dataset.sh [task_type]
# convert to mindrecord
bash scripts/convert_dataset.sh [task_type]
# task type: match, match_kn, match_kn_gene
# default: match_kn_gene
```


***********************************************************************
#####第三部分方法一：
***********************************************************************
#### GPU处理器上运行训练（这里会训练模型，生成save_model文件夹，里面是生成的checkpoint）
```bash
bash scripts/run_train.sh [task_type]
# task type: match, match_kn, match_kn_gene
# default: match_kn_gene
```
#### GPU处理器上预测（利用上一步生成的模型进行预测，生成score.txt和predict.txt）
方法一：
```bash
bash scripts/run_predict.sh
# task type: match, match_kn, match_kn_gene
# default: match_kn_gene
```
方法二：
如果方法一无法直接运行，请尝试方法二：
1、新建一个output空文件夹；
2、一次运行scripts/run_predict.sh里面的每一条命令，需要注意的是load的模型需要换成训练生成的模型；
3、依次执行完后会得到score.txt和predict.txt。

***********************************************************************
#####第三部分方法二：
***********************************************************************
在配置和用法一一样的环境和数据集情况下；
执行jupyter.py即可得到predict.txt和score.txt.

## Citing
### Bibtex

```bibtex
@misc{DuConv_mindspore,
  auther = {lvyufeng},
  year = {2021},
  publisher = {Github}
  journal = {DuConv_mindspore},
  howpublished = {\url{https://github.com/lvyufeng/DuConv_mindspore.git}}
  }
  

