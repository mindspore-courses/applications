<div align=center>
  <h1>Applications</h1>
  <p><a href="./README.md">View English</a></p>
</div>

本仓库提供了系列精心整理的基于 MindSpore 实现的应用案例，覆盖计算机视觉、自然语言处理、生成对抗网络、扩散模型、大语言模型、检索增强生成及智能体等领域。所有案例按领域和模型分类归档，为各类代表性任务提供可复用的实践方案。

## 📢 最新消息

- 2025-11-18 [功能优化]：重构仓库结构以优化应用导航体验；新增Issue与PR模板，让贡献流程更标准化。

## 前置知识

在学习本门课程之前，您需要掌握：

- Python基础
- Linux命令基础
- Jupyter基础
- Docker镜像使用

您可以通过前置学习考试进行自检。

## 环境准备

为确保项目仓中实践代码可正常运行，推荐以下环境准备方式。更多详细的环境准备指南详见[Wiki](https://github.com/mindspore-courses/applications/wiki/Set-Up-Development-Environment)。

### 直接安装依赖

请先确保 Python 版本符合[课程要求](#版本维护)后，进入仓库根目录，执行：

```bash
pip install requirements.txt
```

### 使用Docker镜像（待发布）

为方便开发者更加便捷地进行代码实践，节约环境准备的时间，我们提供了预装好的基础Dockerfile文件。课程的所有镜像可从[dockerfile](./dockerfile/)获取。

镜像基础使用教程详见环境准备Wiki中的[Docker镜像使用](https://github.com/mindspore-courses/applications/wiki/Set-Up-Development-Environment)部分。

## 案例清单

应用案例（通常以 Notebooks 形式呈现）按技术领域分类，各领域下再按模型进一步细分，为开发者提供清晰的索引导航。

| 序号 | 领域     | 描述                           |
| :--- | :------ | :----------------------------- |
| 1 | [CV](./cv/) | 计算机视觉模型及相关任务（图像分类、目标检测、语义分割等）。 |
| 2 | [NLP](./nlp/) | 文本处理、序列建模及语言理解类任务。 |
| 3 | [GAN](./gan/) | 用于图像合成、风格迁移的GAN系列模型。 |
| 4 | [Audio](./audio/) | 音频分类、语音相关任务及信号处理应用案例。 |
| 5 | [Diffusion](./diffusion/) | 基于扩散模型的图像生成及训练流程示例。 |
| 6 | [LLM](./llm/) | 适用于文本生成、逻辑推理及指令遵循类任务的大语言模型应用案例。 |
| 7 | [Multi-Modal](./multi-modal/) | 融合文本、视觉或音频等多种模态的模型应用案例。 |
| 8 | [OrangePi](https://github.com/mindspore-courses/orange-pi-mindspore) | 基于 MindSpore 在 OrangePi 开发板上实现的训推应用案例。 |
| 9 | [RAG](./rag/) | 检索增强生成（RAG）技术的落地案例，含知识库问答、专业文档解析等场景的实现流程与示例。 |
| 10 | [Agent](./agent/) | 智能体（Agent）应用案例，覆盖任务拆解、工具调用类场景的架构工作流与系统实现。 |

## 版本维护

项目随昇思MindSpore及昇思MindSpore NLP套件迭代同步发布版本。

| 版本名  | Python | MindSpore | MindSpore NLP |
| :----- | :----- |:------ |:------ |
| dev   | >=3.9, <=3.11 | 2.7.0    | 0.5.1    |

## 常见问题（FAQ）

详见Wiki中[FAQ](https://github.com/mindspore-courses/applications/wiki/Developer-FAQ)。

## 贡献与反馈

1. **Issue**：欢迎各位开发者通过 [Issue](https://github.com/mindspore-lab/orange-pi-mindspore/issues) 提交建议或 bug 反馈

2. **Pull Request**: 开发者可发起 [PR](https://github.com/mindspore-courses/applications/pulls) 进行Bug修复或代码贡献（提交前请参考[提交规范](https://github.com/mindspore-lab/orange-pi-mindspore/wiki/Contributing-Guidelines)，由Committer @xing-yiren 及另一位Committer 完成评审合入），你的每一份参与都能让本项目更加完善。

3. **开源项目**：若开发者有符合条件的开源项目推荐/自荐，欢迎按照[项目仓规范](https://github.com/mindspore-lab/orange-pi-mindspore/wiki/Contributing-Guidelines)完善项目内容后，邮件至[contact@public.mindspore.cn](mailto:contact@public.mindspore.cn)进行投稿，邮件标题请参考：`【昇思+香橙派项目投稿】项目名称`格式，并在正文中对项目进行简单介绍，附上代码仓链接。

### 提交规范

详见WIKI：[Issue与PR提交规范](https://github.com/mindspore-courses/applications/wiki/Contributing-Guidelines)

### 贡献者展示

向本项目的贡献者们致以最诚挚的感谢！

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/mindspore-courses/applications/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=mindspore-courses/applications" />
  </a>
</div>
