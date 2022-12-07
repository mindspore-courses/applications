# 基于MindSpore框架的MASS案例实现
## 1 模型简介
微软亚洲研究院于2019在ICML发表《MASS: Masked Sequence to Sequence Pre-training for Language Generation》，其借鑑了Bert的Masked Language Model预训练任务，提出了MAsked Sequence to Sequence Pre-training（MASS）模型，为自然语言生成任务联合预训练编码器和解码器。

MASS的编码器-解码器结构示例，图中“\_”表示被屏蔽的词。

<div align=center>
<img src='https://i.imgur.com/Jvhm0Dx.png' width='600px'>
</div>

编码器： 以被随机屏蔽掉连续片段的句子作为输入，BERT的做法是随机屏蔽掉15%的词，而MASS为了解决编码与解码之间的平衡，做法为屏蔽掉句子总长50%的片段。模型中使用特殊符号 $[\mathbb M]$ 替换连续的单词来屏蔽片段，起始位置是随机的，且被选中的token有80%的概率是正常的 $[\mathbb M]$ token，10%的概率是被随机token替换，10%的概率保持原来的token。以上图为例，其中输入序列有8个单词，片段 $x_3-x_6$ 被屏蔽掉。

解码器：输入为与编码器同样的序列，但是会屏蔽掉剩馀的词，然后解码器只预测编码器端屏蔽掉的词。以上图为例，只给定 $x_3x_4x_5$ 作为位置 4 - 6 的解码器输入，解码器会将 $[\mathbb M]$ 作为其他位置的输入（屏蔽了位置 1 − 3 和 7 − 8）。为了减少内存和计算成本，被屏蔽的token会被移除，未屏蔽token的位置编码不变（如果前两个标记被屏蔽并移除，第三个标记的位置仍然是 2 而不是 0)。通过这种方式，可以获得相似的准确度，并在解码器中减少 50% 的计算量。


```
encoder input (source): [x1, x2, x3, x4, x5, x6, x7, x8, </eos>]
masked encoder input:   [x1, x2, x3,  _,  _,  _, x7, x8, </eos>]
decoder input:          [  -, x3, x4, x5]
                          |   |   |   |
                          V   V   V   V
decoder output:         [x3, x4, x5, x6]
```

MASS预训练有以下几大优势：

(1) 编码器被强制去抽取未被屏蔽掉的词的含义，可以提升编码器理解源序列文本的能力。<br>
(2) 通过在解码器端预测连续的标记，解码器可以比仅预测离散标记拥有更好的语言建模能力。<br>
(3) 通过在解码器端进一步屏蔽在编码器端未被屏蔽掉的词， 以鼓励解码器从编码器端提取更多有用的信息来做预测，而不是依赖于前面预测出的单词，这样能促进编码器-解码器结构的联合训练。

### 1.1 模型结构

其模型基础结构可以使用任何Seq2Seq的结构，由于Transformer的优越性，故论文中使用Transformer模型作为基础结构，Transformer整体架构由编码器和解码器两个部分组成，不依赖任何RNN和CNN结构来生成输出，而是使用了Attention注意力机制，自动学习输入序列中每个单词和其他单词的关联，可以更好的处理长文本，且该模型可以高效的并行工作，训练速度较快。

Transformer 的整体架构如下：

<div align=center>
<img src='https://i.imgur.com/ooO7ULP.png' width='400px'>
</div>

- 编码器和解码器分别由 $N=6$ 个相同的编码器/解码器层组成。
- 在 Transformer 架构的左半部分，编码器的任务是将输入序列映射到一系列连续表示，然后将其馈送到解码器。
- 架构右半部分的解码器接收编码器的输出以及前一个时间步的解码器输出，以生成输出序列。
- 解码器的输出最终通过一个全连接层，然后是一个 softmax 层，以生成对输出序列下一个单词的预测。

### 1.2目标函数

给定一个未配对的源句子 $x ∈ \mathcal X$，MASS通过被屏蔽的序列 $x^{\setminus u:v}$ 作为输入来预测句子片段 $x^{u:v}$ 以预训练序列到序列模型。目标函数为一极大似然函数：

```math
L(\theta; \mathcal X) = \frac{1}{|\mathcal X|} \sum_{x \in \mathcal X} \log P(x^{u:v} | x^{\setminus u:v}; \theta) = \frac{1}{|\mathcal X|} \log \Pi^{v}_{t=u} P(x^{u:v}_{t}|x^{u:v}_{\textless t}, x^{\setminus u:v};\theta)
```

注: $x^{u:v}$ 表示以句子位置 $u$ 为起点 $v$ 为终点的片段； $x^{\setminus u:v}$ 为 $x^{u:v}$ 的修改版本，从 $u$ 到 $v$ 的片段被屏蔽， $0 < u < v < m$ 其中 $m$ 是句子 $x$ 长度。

### 1.3 模型特点
MASS 有一个重要的超参数 $k$，表示屏蔽的连续片段长度，通过调整 $k$ 的大小，MASS 能包含 BERT 中的掩码语言模型训练方法以及 GPT 中标准的语言模型预训练方法，使 MASS 成为一个通用的预训练框架。

当 $k = 1$ 时，根据MASS的设定，编码器端仅屏蔽一个单词，解码器以源序列中未屏蔽的单词为条件预测这个单词，如图(a)所示。由于解码器的所有输入都被屏蔽了，因此解码器本身就像一个非线性分类器，类似于 BERT 中使用的 softmax 矩阵。在这种情况下，条件概率为 $P (x^u|x^{\setminus u}; θ)$， $u$ 是掩码标记的位置，这正是 BERT3中使用的掩码语言模型的公式。

当 $k = m$（ $m$ 为序列长度）时，根据MASS的设定，编码器会屏蔽所有的单词，解码器需要预测所有单词，如图(b)所示。由于编码器端所有词都被屏蔽了，解码器的注意力机制相当于没有获取到信息，在这种情况下条件概率为 $P(x^{1:m}|x^{\setminus 1:m}; θ)$，等价于GPT中的标准语言模型。

![](https://i.imgur.com/JdTGb9z.png)

## 2 代码结构

MASS脚本及代码结构如下：

```text
├── mass
  ├── config
  │   ├──config.py                           // 参数配置
  ├── src
      │   ├──model_utils
      │   ├──config.py                       // 参数配置
      │   ├──device_adapter.py               // 设备配置
      │   ├──local_adapter.py                // 本地设备配置
      │   ├──moxing_adapter.py               // modelarts设备配置
  ├──src
  │   ├──dataset
  │      ├──bi_data_loader.py                // 数据集加载器，用于微调或推理
  │      ├──mono_data_loader.py              // 预训练数据集加载器
  │   ├──language_model
  │      ├──noise_channel_language_model.p   // 数据集生成噪声通道语言模型
  │      ├──mass_language_model.py           // 基于MASS论文的MASS语言模型
  │      ├──loose_masked_language_model.py   // 基于MASS发布代码的MASS语言模型
  │      ├──masked_language_model.py         // 基于MASS论文的MASS语言模型
  │   ├──transformer
  │      ├──create_attn_mask.py              // 生成屏蔽矩阵，除去填充部分
  │      ├──transformer.py                   // Transformer模型架构
  │      ├──encoder.py                       // Transformer编码器组件
  │      ├──decoder.py                       // Transformer解码器组件
  │      ├──self_attention.py                // 自注意块组件
  │      ├──multi_head_attention.py          // 多头自注意组件
  │      ├──embedding.py                     // 嵌入组件
  │      ├──positional_embedding.py          // 位置嵌入组件
  │      ├──feed_forward_network.py          // 前馈网络
  │      ├──residual_conn.py                 // 残留块
  │      ├──beam_search.py                   // 推理所用的波束搜索解码器
  │      ├──transformer_for_infer.py         // 使用Transformer进行推理
  │      ├──transformer_for_train.py         // 使用Transformer进行训练
  │   ├──utils
  │      ├──byte_pair_encoding.py            // 使用subword-nmt应用字节对编码（BPE）
  │      ├──dictionary.py                    // 字典
  │      ├──loss_moniter.py                  // 训练步骤中损失监控回调
  │      ├──lr_scheduler.py                  // 学习速率调度器
  │      ├──ppl_score.py                     // 基于N-gram的困惑度评分
  │      ├──rouge_score.py                   // 计算ROUGE得分
  │      ├──load_weights.py                  // 从检查点或者NPZ文件加载权重
  │      ├──initializer.py                   // 参数初始化器
  ├── vocab
  │   ├──all.bpe.codes                       // 字节对编码表
  │   ├──all_en.dict.bin                     // 已学习到的词汇表
  ├── scripts
  │   ├──run_ascend.sh                       // Ascend处理器上训练&评估模型脚本
  │   ├──run_gpu.sh                          // GPU处理器上训练&评估模型脚本
  │   ├──learn_subword.sh                    // 学习字节对编码
  │   ├──stop_training.sh                    // 停止训练
  ├── requirements.txt                       // 第三方包需求
  ├── train.py                               // 训练API入口
  ├── eval.py                                // 推理API入口
  ├── default_config.yaml                    // 参数配置
  ├── tokenize_corpus.py                     // 语料标记化
  ├── apply_bpe_encoding.py                  // 应用BPE进行编码
  ├── weights_average.py                     // 将各模型检查点平均转换到NPZ格式
  ├── news_crawl.py                          // 创建预训练所用的News Crawl数据集
  ├── gigaword.py                            // 创建Gigaword语料库
```
