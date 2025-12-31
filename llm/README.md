# Large Language Model Applications

This directory contains ready-to-use Large Language Model application notebooks built with MindSpore. Each notebook demonstrates a complete or partial workflow—training, finetuning, or inference—along with a brief introduction to the model used.

## Application List

### Actively Maintained Applications

The following notebooks are actively maintained in sync with MindSpore and MindSpore NLP releases and will continue to evolve as the framework advances.

| No. | Model | Description              |
| :-- | :---- | :----------------------- |
| 1   | [t5](./t5/) | Includes notebooks for T5 finetuning and inference on tasks such as email summarization |

### Community-Driven / Legacy Applications

Addtional community-contributed or legacy notebooks are stored under the [legacy](./legacy/) directory. These notebooks are not actively maintained and may rely on older APIs.

If a notebook does not specify environment requirements, assume it runs with:

- mindspore == 2.3.1
- mindnlp == 0.4.1

> If you need any of these applications updated, please open an [Issue](https://github.com/mindspore-courses/applications/issues) with the required MindSpore and MindSpore NLP version.
> Developers are also welcome to update these notebooks to the latest version by submitting a [Pull Request](https://github.com/mindspore-courses/applications/pulls).

## Contributing New LLM Applications

To contribute a new LLM application:

1. Place your notebook in the corresponding model directory.
2. If the model does not yet have its own directory, create a new one following the existing structure.
3. Follow the notebook writing and naming standards in the [Contributing Guidelines](https://github.com/mindspore-courses/applications/wiki/Contributing-Guidelines).
4. Update the application list in the README if required.
