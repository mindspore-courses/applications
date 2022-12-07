# 项目介绍

​	DuConv是百度发布的基于知识图谱的主动聊天任务数据集，让机器根据构建的知识图谱进行主动聊天，使机器具备模拟人类用语言进行信息传递的能力，发表于ACL2019。通过使用一个额外的知识选择范式增强了普通的seq2seq模型，来生成知识驱动的对话响应。

​     	DuConv 数据集和基于 DuConv 的知识感知模型是百度于2019年为了开发具有主动引导能力的对话系统所创建的数据集和网络模型。基于DuConv的知识感知模型分为基于检索的模型和基于生成的模型。Retrieval-based 模型中的编码器具有与Bert相同的结构，并且引入双向GPU的向量表示，使用Transformer方法进行编码表示，再联合attention方法选取的相关知识信息通过softmax层进行二分类判断。Generation-based知识感知模型通过引入先验分布和后验分布，同时引入attention机制以及使用两种不同的损失函数来分别衡量模型输出和真实输出差异、真实输出和相外部知识的差异。与传统的seq2seq模型相比，Retrieval-based 模型和Generation-based 模型都具有更好的性能，可以利用DuConv数据集中更多的知识图谱来实现引导对话的目标，并且同时保证生成的对话自然连贯。本次应用案例实现的是基于DuConv 数据集的Retrieval-based 模型。



# 环境依赖

​	版本：MindSpore 1.8.1 + python 3.7

​	GPU 环境：CUDA11.1+Nvidia Tesla V100-SXMZ 或 Nvidia CUDA11.1+GeForce RTX 3090Ascend 

​	环境：Ascend 910 CPU24核 内存96GiB MindSpore 1.8.1



# 目录结构描述
    ├── ReadMe.md                   // 帮助文档
    
    ├── DuConv_DataProcess.ipynb    // 用来进行数据预处理的ipynb文件
    
    ├── DuConv_train_predict.ipynb  // 用来进行模型训练和预测的ipynb文件
    
    ├── data                        // 存放数据的文件夹，其中包括网上下载的txt文件，处理的过程文件以及最                                    终生成的mindrecord文件
    
    ├── output                      // 存放预测的语句结果和预测评分的文件夹
    
    ├── save_model                  // 存放模型的文件夹
    
    ├── kernel_meta                 // 过程性调试文件夹
    
    ├── rank_0                      // 过程性调试文件夹



# 使用说明

1、运行之前请先自己删除data、kernel_meta、output、rank_0和save_model文件夹，只保留俩个ipynb文件和说明文档

2、首先逐步运行DuConv_DataProcess.ipynb文件来进行数据预处理，从而将gz类型的数据集下载到本地并进行处理，最终生成mindrecord类型的训练集、验证集和测试集。

3、逐步运行DuConv_train_predict.ipynb文件来完成模型的训练和预测，训练完成的模型存在save_model目录下，预测出的结果以及评分存在output目录下。最终对预测的结果的各项指标进行评估，并将评估结果打印出来。



