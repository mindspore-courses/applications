# “心晴有答”——基于MindFormers的智能情感陪伴小助手

#### XinQing - Mental health support application

<img src=".\pic\logo.png" alt="logo" style="zoom:33%;" />

## 项目简介

本项目主要需要实现智能心理咨询问答的功能，并将用户与机器人的聊天记录存储到本地，整理成结构化的日记数据，对用户的心理健康状况进行追踪，并参考 CBT 给出相应的练习（五栏表填写）帮助用户维持良好的心理状态，且能够通过聊天文本对用户进行智能的心理辅导，必要时进行心理危机干预。

运行客户端，你可以看到心晴的自我介绍：

##### 哈喽！我是心晴😊

```
你好，很高兴能与你相识。

我是由华南师范大学盲盒开金光队精心打造的情感陪伴机器人。我的主要功能是与你进行深入的交流，为你提供情感上的支持和心理疏导。

我拥有丰富的情感交流经验，能够准确地理解你的情绪状态，并提供相应的支持和建议。

无论你是感到孤独、沮丧还是焦虑，我都将竭尽全力帮助你度过难关，在你需要的时候始终陪伴在你身边🌈
```

## 快速开始

1. 配置环境
   `pip install -r requirements.txt`

2. 运行交互系统

   `python server.py` 

   `chainlit run ./web/app.py -w`

## 模型训练

- 本项目基于 Mindformers 的 ChatGLM-6B 模型进行 LoRA 微调

- 训练过程中通过 data2indrecord.py 对数据集进行格式转换，使用生成的 smile_train.mindrecord 数据进行训练

- 使用 task.py 执行低参微调实现 LoRA 微调效果


参考框架：MindFormers、MindNLP、MindSpore

## 引用

项目采用MeChat使用的SMILE数据集：

```tex
@misc{qiu2023smile,
      title={SMILE: Single-turn to Multi-turn Inclusive Language Expansion via ChatGPT for Mental Health Support},
      author={Huachuan Qiu and Hongliang He and Shuai Zhang and Anqi Li and Zhenzhong Lan},
      year={2023},
      eprint={2305.00450},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

## 免责声明

心理咨询机器人“心晴”旨在为用户提供心理健康方面的支持和指导。然而，机器人并不能替代专业心理医生的诊断和治疗。如果您有严重的心理问题或需要专业的心理治疗，请寻求专业医生的帮助。

此外，心理咨询机器人“心晴”所提供的建议和信息仅供参考，不应被视为最终的解决方案。用户应根据自身情况和需要，谨慎考虑并自行决定是否采纳机器人的建议。机器人不会为用户的决定和行为承担任何责任。

同时，心理咨询机器人“心晴”在提供服务时，可能会涉及到用户的个人隐私和敏感信息。机器人将严格遵守相关法律法规，采取必要的措施保护用户的隐私和安全。

最后，心理咨询机器人“心晴”会尽力为用户提供优质的服务，但对于因不可抗力或其他原因导致的服务中断或问题，机器人不承担责任。如果您在使用过程中遇到任何问题或困惑，可以随时联系我们，我们将尽力协助您解决问题。

使用心理咨询机器人“心晴”，视为您已经仔细阅读并理解本免责声明的全部内容。如果您不同意本声明的内容，请不要使用机器人服务。

## 项目结构

下面是本项目的结构：

```
│  README.md
│  requirements.txt
│  server.py
│  tree.txt
│  技术报告（盲盒开金光队）.pdf
│  
├─developing
│  ├─local_applet
│  │  │  .eslintrc.js
│  │  │  app.js
│  │  │  app.json
│  │  │  app.wxss
│  │  │  project.config.json
│  │  │  project.private.config.json
│  │  │  sitemap.json
│  │  │  
│  │  └─pages
│  │      ├─image
│  │      │      image.js
│  │      │      image.wxml
│  │      │      心情记录.png
│  │      │      心理测试.png
│  │      │      心理科普.png
│  │      │      我的.png
│  │      │      智能对话.png
│  │      │      滚动图片1.jpg
│  │      │      聊天.png
│  │      │      首页.png
│  │      │      
│  │      ├─index
│  │      │      index.js
│  │      │      index.json
│  │      │      index.wxml
│  │      │      index.wxss
│  │      │      
│  │      ├─my
│  │      │      my.js
│  │      │      my.json
│  │      │      my.wxml
│  │      │      my.wxss
│  │      │      
│  │      └─talk
│  │              talk.js
│  │              talk.json
│  │              talk.wxml
│  │              talk.wxss
│  │              
│  └─local_web
│      │  .gitignore
│      │  index.html
│      │  README.md
│      │  
│      ├─css
│      │      1
│      │      aboutus
│      │      calender
│      │      reset.css
│      │      style.css
│      │      
│      ├─img
│      │      cd-arrow.svg
│      │      cd-logo.svg
│      │      cd-socials.svg
│      │      download.png
│      │      favicon.ico
│      │      left.png
│      │      man.png
│      │      msg-input.png
│      │      woman.png
│      │      
│      ├─js
│      │      chat.js
│      │      city.js
│      │      flexible.js
│      │      index.js
│      │      jquery.min.js
│      │      main.js
│      │      modernizr.js
│      │      picker.min copy.js
│      │      picker.min.js
│      │      
│      ├─partials
│      │      _layout.scss
│      │      _mixins.scss
│      │      _variables.scss
│      │      
│      └─scss
│              style.scss
│              
├─src
│  │  lora.py
│  │  task.py
│  │  
│  ├─checkpoints
│  │  │  glm_6b_chat.ckpt
│  │  │  
│  │  └─.ipynb_checkpoints
│  │          glm_6b_chat-checkpoint.yaml
│  │          
│  ├─data_processing
│  │  │  data2mindrecord.py
│  │  │  ice_text.model
│  │  │  
│  │  └─train
│  │          smile_train.mindrecord
│  │          smile_train.mindrecord.db
│  │          
│  ├─langchain_test
│  │  │  chatbot.py
│  │  │  glm_with_huggingface_embdeding.py
│  │  │  glm_with_langchain.py
│  │  │  
│  │  ├─scrs
│  │  │      chinese_test_spliter.py
│  │  │      ErnieModel.py
│  │  │      msembeding.py
│  │  │      mspipeline.py
│  │  │      msTransformers.py
│  │  │      __init__.py
│  │  │      
│  │  └─utils
│  │          model_convert.py
│  │          vocab_convert.py
│  │          
│  └─mindpet
│      │  .gitignore
│      │  LICENSE
│      │  README.md
│      │  set_up.py
│      │  
│      ├─doc
│      │  │  TK_DeltaAlgorithm_README.md
│      │  │  TK_GraphOperation_README.md
│      │  │  
│      │  └─image
│      │          architecture_of_adapter_module.png
│      │          architecture_of_low_rank_adapter_module.png
│      │          lora.PNG
│      │          prefix.png
│      │          
│      ├─test
│      │  ├─developer_test
│      │  │  ├─exit_code
│      │  │  │      launcher_demo.py
│      │  │  │      
│      │  │  └─task
│      │  │          eval_launcher.py
│      │  │          finetune_launcher.py
│      │  │          infer_launcher.py
│      │  │          model_config_eval.yaml
│      │  │          model_config_finetune.yaml
│      │  │          model_config_infer.yaml
│      │  │          
│      │  └─unit_test
│      │      │  run_test.sh
│      │      │  
│      │      ├─delta
│      │      │      test_adapter.py
│      │      │      test_lora.py
│      │      │      test_low_rank_adapter.py
│      │      │      test_prefix_layer.py
│      │      │      test_rdrop.py
│      │      │      
│      │      ├─entry
│      │      │  │  test_tk_main.py
│      │      │  │  test_tk_sdk.py
│      │      │  │  
│      │      │  └─resource
│      │      │      ├─boot_file
│      │      │      │      boot_evaluate.py
│      │      │      │      boot_finetune.py
│      │      │      │      boot_infer.py
│      │      │      │      
│      │      │      └─model_config
│      │      │              model_config_evaluate_infer.yaml
│      │      │              model_config_finetune.yaml
│      │      │              
│      │      ├─graph
│      │      │  │  test_freeze_utils.py
│      │      │  │  test_save_ckpt.py
│      │      │  │  
│      │      │  ├─data
│      │      │  │      train-images.idx3-ubyte
│      │      │  │      train-labels.idx1-ubyte
│      │      │  │      
│      │      │  ├─output
│      │      │  │      checkpoint_base-1_3.ckpt
│      │      │  │      checkpoint_base-graph.meta
│      │      │  │      
│      │      │  └─resource
│      │      │          test_freeze_config_file.yaml
│      │      │          test_freeze_config_file_include_not_list.yaml
│      │      │          test_freeze_config_file_no_freeze_key.yaml
│      │      │          test_freeze_config_file_no_include_and_exclude.yaml
│      │      │          
│      │      ├─log
│      │      │      test_concurrent_log.py
│      │      │      test_logger.py
│      │      │      test_logger_validator.py
│      │      │      
│      │      ├─security
│      │      │  └─param_check
│      │      │          test_base_check.py
│      │      │          test_model_config_check_utils.py
│      │      │          test_option_check_utils.py
│      │      │          
│      │      ├─task
│      │      │  ├─evaluate_infer
│      │      │  │  │  .test_result_file_check.py.swp
│      │      │  │  │  test.sh
│      │      │  │  │  test_evaluate_infer_task.py
│      │      │  │  │  test_result_file_check.py
│      │      │  │  │  
│      │      │  │  └─resource
│      │      │  │          boot_file_without_generating_result_json.py
│      │      │  │          boot_file_with_empty_json.py
│      │      │  │          boot_file_with_link_result_json.py
│      │      │  │          boot_file_with_oversize_result_json.py
│      │      │  │          boot_file_with_runtime_error.py
│      │      │  │          normal_evaluate_boot_file.py
│      │      │  │          normal_infer_boot_file.py
│      │      │  │          
│      │      │  └─finetune
│      │      │          test_finetune_options_check.py
│      │      │          test_finetune_task.py
│      │      │          
│      │      └─utils
│      │          │  test_entrance_monitor.py
│      │          │  test_io_utils.py
│      │          │  test_task_utils.py
│      │          │  
│      │          └─resource
│      │                  timeout_monitor_task.py
│      │                  
│      └─tk
│          │  tk_main.py
│          │  tk_sdk.py
│          │  __init__.py
│          │  
│          ├─delta
│          │      adapter.py
│          │      delta_constants.py
│          │      lora.py
│          │      low_rank_adapter.py
│          │      prefix_layer.py
│          │      r_drop.py
│          │      __init__.py
│          │      
│          ├─graph
│          │      ckpt_util.py
│          │      freeze_utils.py
│          │      __init__.py
│          │      
│          ├─log
│          │      log.py
│          │      log_utils.py
│          │      __init__.py
│          │      
│          ├─security
│          │  │  __init__.py
│          │  │  
│          │  └─param_check
│          │          base_check.py
│          │          model_config_params_check_util.py
│          │          option_check_utils.py
│          │          __init__.py
│          │          
│          ├─task
│          │  │  option_decorators.py
│          │  │  __init__.py
│          │  │  
│          │  ├─evaluate_infer
│          │  │      evaluate_infer_task.py
│          │  │      result_file_check.py
│          │  │      __init__.py
│          │  │      
│          │  ├─finetune
│          │  │      finetune_options_check.py
│          │  │      finetune_task.py
│          │  │      __init__.py
│          │  │      
│          │  └─options
│          │          boot_file_path_option.py
│          │          ckpt_path_option.py
│          │          data_path_option.py
│          │          model_config_path_option.py
│          │          output_path_option.py
│          │          path_check_param.py
│          │          pretrained_model_path_option.py
│          │          quiet_option.py
│          │          timeout_option.py
│          │          __init__.py
│          │          
│          └─utils
│                  constants.py
│                  entrance_monitor.py
│                  exceptions.py
│                  io_utils.py
│                  task_utils.py
│                  version_control.py
│                  version_utils.py
│                  __init__.py
│                  
└─web
    │  .dockerignore
    │  app.py
    │  chainlit.md
    │  Procfile
    │  requirements.txt
    │  
    ├─.chainlit
    │      .langchain.db
    │      config.toml
    │      
    ├─static
    │      README.md
    │      
    └─__pycache__
            app.cpython-39.pyc
```
