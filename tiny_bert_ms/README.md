# tiny_bert_ms
tiny_bert on mindspore

介绍了Bert与TinyBert 并在mindspore上运行。


(注意项目代码目前是在Ascend平台运行的代码， 如果需要在GPU运行， 需要在代码的 预训练蒸馏config部分 和 任务蒸馏config部分 
手动修改device_target ="GPU"  

ipynb文件可以端到端运行， 但需要提前下载好预训练模型和训练数据。 在bert文件夹下，缺少由tensorflow转换来的模型。
ms2tf包含了转换的代码，具体模型下载转换和数据下载操作见tinybert.ipynb的模型准备部分。

下面是路径结构

```
.tiny_bert_ms/
└── bert                                    # 模型相关文件代码。
    ├── ms2tf                                     #存放将tf的bert模型转ms模型的代码
    ├── uncased_L-12_H-768_A-12                   # 存放tf模型相关文件，包含词表和config
    └── tf2ms.ipynb                               # tf转ms的操作代码。 需要环境中存在tf和ms
    
└── config                                  # 训练设置的yaml文件。
└── data                                    # 训练所需数据
└── ipyphoto                                # ipy文件中的图片
└── tinybert.ipynb                          # 主体代码， 包含原理介绍与代码解析。


```

## citing
### BibTex


@article{jiao2019tinybert,
  title={Tinybert: Distilling bert for natural language understanding},
  author={Jiao, Xiaoqi and Yin, Yichun and Shang, Lifeng and Jiang, Xin and Chen, Xiao and Li, Linlin and Wang, Fang and Liu, Qun},
  journal={arXiv preprint arXiv:1909.10351},
  year={2019}
}
