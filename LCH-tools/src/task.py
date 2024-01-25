import argparse

from mindformers import Trainer, TrainingArguments
from mindformers import init_context, ContextConfig, ParallelContextConfig

def context_init(use_parallel=False, optimizer_parallel=False):
    """init context for mindspore."""
    context_config = ContextConfig(mode=0, device_target="Ascend", device_id=0)
    parallel_config = None
    if use_parallel:
        parallel_config = ParallelContextConfig(parallel_mode='SEMI_AUTO_PARALLEL',
                                                gradients_mean=False,
                                                enable_parallel_optimizer=optimizer_parallel,
                                                full_batch=True)
    rank_id, device_num = init_context(use_parallel=use_parallel,
                                       context_config=context_config,
                                       parallel_config=parallel_config)

def main(use_parallel=False,
         run_mode='train',
         task='text_generation',
         model_type='glm_6b',
         checkpoint_path='./glm_6b.ckpt',
         train_dataset='./train',
         eval_dataset='./eval',
         predict_data='你好',
         batch_size=4,
         dp=1, mp=1, pp=1, micro_size=1, op=False):
    if use_parallel.lower() == "true":
        use_parallel = True
    else:
        use_parallel = False

    # 环境初始化
    context_init(use_parallel, op)
    # 训练超参数定义
    training_args = TrainingArguments(num_train_epochs=1, batch_size=batch_size, learning_rate=5e-5, warmup_steps=100, sink_mode=True, sink_size=4)
    # 定义任务，预先准备好相应数据集
    task = Trainer(task=task, model=model_type, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset)
    task.set_parallel_config(data_parallel=dp,
                             model_parallel=mp,
                             pipeline_stage=pp,
                             optimizer_shard=op,
                             micro_batch_num=micro_size)

    if run_mode == 'train':
        # 训练
        task.train()
    elif run_mode == 'finetune':
        # 微调
        task.finetune(checkpoint_path)
    elif run_mode == 'eval':
        # 评估
        task.evaluate(checkpoint_path)
    elif run_mode == 'predict':
        # 推理，仅支持单卡推理
        assert use_parallel == False, "only support predict under stand_alone mode."
        result = task.predict(input_data=predict_data)
        print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_mode', default='train', required=True, help='set run mode for model.')
    parser.add_argument('--use_parallel', default=False, help='open parallel for model.')
    parser.add_argument('--task', default='text_generation', required=True, help='set task type.')
    parser.add_argument('--model_type', default='glm_6b', required=True, help='set model type.')
    parser.add_argument('--checkpoint_path', default=None, help='set checkpoint path.')
    parser.add_argument('--train_dataset', default=None, help='set train dataset.')
    parser.add_argument('--eval_dataset', default=None, help='set eval dataset.')
    parser.add_argument('--batch_size', default=4, help='batch size of dataset.')
    parser.add_argument('--data_parallel', default=1, type=int,help='set data parallel number. Default: None')
    parser.add_argument('--model_parallel', default=1, type=int, help='set model parallel number. Default: None')
    parser.add_argument('--pipeline_parallel', default=1, type=int, help='set pipeline parallel number. Default: None')
    parser.add_argument('--micro_size', default=1, type=int, help='set micro batch number. Default: None')
    parser.add_argument('--optimizer_parallel', default=False, type=bool, help='whether use optimizer parallel. Default: None')
    args = parser.parse_args()
    print(args)

    main(run_mode=args.run_mode,
         task=args.task,
         use_parallel=args.use_parallel,
         model_type=args.model_type,
         checkpoint_path=args.checkpoint_path,
         train_dataset=args.train_dataset,
         eval_dataset=args.eval_dataset,
         batch_size=int(args.batch_size),
         dp=args.data_parallel,
         mp=args.model_parallel,
         pp=args.pipeline_parallel,
         micro_size=args.micro_size,
         op=args.optimizer_parallel)
         