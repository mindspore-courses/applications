<div align=center>
  <h1>Applications</h1>
  <p><a href="./README_ZH.md">查看中文</a></p>
</div>

This repository provides a curated collection of AI application examples built with **MindSpore**, covering domains such as CV, NLP, GANs, Diffusion Models, LLMs, RAG, and Agents. Each case is organized by **domain** and **model**, offering reusable implementations for representative tasks. It serves as a practical reference for developers to explore and extend AI solutions within the MindSpore ecosystem.

## 📢 News

- **2025-11-18 [Feature Optimization]**：Repository refactored for clearer application navigation; added Issue and Pull Request templates for more standardized contributions.

## Prerequisites

Before starting this course, you should be familiar with:

- Basic Python programming
- Basic Linux commands
- Using Jupyter Notebook
- Using Docker images

You can take the Prerequisite Test to assess your readiness.

## Environment Setup

To ensure all example code runs smoothly, set up your environment using one of the following methods.

### Install Dependencies

Confirm your Python version meets the course requirements, then run:

```bash
pip install -r requirements.txt
```

### Use Docker Image (*Coming Soon*)

Prebuilt Dockerfiles are provided to simplify environment setup.

You can find all course images in the [dockerfile](./dockerfile/) directory and pull the one that fits your hardware.

For details, see [Using Docker Images](https://github.com/mindspore-courses/applications/wiki/Set-Up-Development-Environment) in Wiki.

## Application List

The notebooks are organized by domain. Within each domain, notebooks are further grouped by model to provide a clear and scalable structure.

| No.  | Domain  | Description                    |
| :--- | :------ | :----------------------------- |
| 1 | [CV](./cv/) | Vision models and tasks (classification, detection, segmentation). |
| 2 | [NLP](./nlp/) | Text processing, sequence modeling, and language understanding tasks. |
| 3 | [GAN](./gan/) | GAN models for image synthesis and style transfer. |
| 4 | [Audio](./audio/) | Audio classification, speech tasks, and signal processing examples. |
| 5 | [Diffusion](./diffusion/) | Diffusion-based generation models and training workflows. |
| 6 | [LLM](./llm/) | Large language models for text generation, reasoning, and instruction tasks. |
| 7 | [Multi-Modal](./multi-modal/) | Models combining text, vision, or audio modalities. |
| 8 | [OrangePi](https://github.com/mindspore-courses/orange-pi-mindspore) | Edge-AI applications on OrangePi with MindSpore. |
| 9 | [RAG](./rag/) | Retrieval-augmented generation pipelines and examples. |
| 10 | [Agent](./agent/) | Agent-style workflows and task-oriented intelligent systems. |

## Awesome projects using MindSpore

- [ChatPDF](https://github.com/lvyufeng/ChatPDF): an application that allows users to upload PDF files and interact with pdf using a chatbot.![GitHub Repo stars](https://img.shields.io/github/stars/lvyufeng/ChatPDF)
- [Emotional-News-Anchor](https://github.com/sunnyxrxrx/Emotional-News-Anchor): an AI-powered news reader that uses LLMs to analyze and broadcast news with emotion-aligned narration.![GitHub Repo stars](https://img.shields.io/github/stars/sunnyxrxrx/Emotional-News-Anchor)

## Version Management

This repository is updated in sync with **MindSpore** and the **MindSpore NLP** Suite.

| Branch/Version  | Python | MindSpore | MindSpore NLP |
| :------ | :----- |:------ |:------ |
| dev   | >=3.9, <=3.11 | 2.7.0    | 0.5.1    |

## FAQ

See the [FAQ](https://github.com/mindspore-courses/applications/wiki/Developer-FAQ) in the Wiki for details.

## Contributing

We welcome bug reports, suggestions, and code contributions via [Issues](https://github.com/mindspore-courses/applications/issues) or [PRs](https://github.com/mindspore-courses/applications/pulls). Please follow our submission guidelines — all PRs are reviewed and merged by @xing-yiren and another committer. Your contributions make the project stronger!

**Guidelines**: [Contributing Guidelines](https://github.com/mindspore-courses/applications/wiki/Contributing-Guidelines)

## Contributors

Special thanks to all contributors for improving this project!

<div align=center style="margin-top: 30px;">
  <a href="https://github.com/mindspore-courses/applications/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=mindspore-courses/applications" />
  </a>
</div>
