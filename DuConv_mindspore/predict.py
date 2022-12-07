import os
import time
import argparse
import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore import context
from mindspore.common.tensor import Tensor
import mindspore.ops.operations as P
from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.model import RetrievalWithSoftmax
from src.bert import BertConfig
from src.dataset import create_dataset
from src.utils.extract import extract_predict_utterance

# def parse_args():
#     """set and check parameters"""
#     parser = argparse.ArgumentParser(description='train duconv')
#     parser.add_argument('--task_name', type=str, default='match_kn', choices=['match', 'match_kn', 'match_kn_gene'],
#                         help='task name for training')
#     parser.add_argument('--max_seq_length', type=int, default=512,
#                         help='Number of word of the longest seqence. (default: %(default)d)')
#     parser.add_argument('--batch_size', type=int, default=8096,
#                         help='Total token number in batch for training. (default: %(default)d)')
#     parser.add_argument('--vocab_size', type=int, default=14373,
#                         help='Total token number in batch for training. (default: %(default)d)')
#     parser.add_argument("--eval_data_file_path", type=str, default="",
#                         help="Data path, it is better to use absolute path")
#     parser.add_argument("--load_checkpoint_path", type=str, default="", help="Save checkpoint path")
#     parser.add_argument("--save_file_path", type=str, default="", help="Save checkpoint path")
#     args = parser.parse_args()
#
#     return args

def run_duconv():
    candidate_file = "data/dandidate.test.txt"
    score_file = "output/score.txt"
    predict_file = "output/predict.txt"
    load_checkpoint_path = "save_model2/match_kn-3_568.ckpt"
    task_name = "match_kn_gene"
    max_seq_length = 128
    batch_size = 1
    eval_data_file_path = "data/test.mindrecord"
    load_checkpoint_path = "save_model2/match_kn-3_568.ckpt"
    save_file_path ="output/score.txt"
    vocab_size = 14373


    """run duconv task"""
    # args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, max_call_depth=10000)
    # use_kn = True if "kn" in args.task_name else False
    # config = BertConfig(seq_length=max_seq_length, vocab_size=vocab_size)
    # dataset = create_dataset(args.batch_size, data_file_path=args.eval_data_file_path,
    #                          do_shuffle=False, use_knowledge=use_kn)
    use_kn = "match_kn_gene"
    # seq_length=max_seq_length
    # vocab_size=vocab_size
    config = BertConfig(seq_length=max_seq_length, vocab_size=vocab_size)
    dataset = create_dataset(batch_size, data_file_path=eval_data_file_path,
                             do_shuffle=False, use_knowledge=use_kn)
    steps_per_epoch = dataset.get_dataset_size()
    print(steps_per_epoch)

    network = RetrievalWithSoftmax(config, use_kn)
    param_dict = load_checkpoint(load_checkpoint_path)
    not_loaded = load_param_into_net(network, param_dict)
    print(not_loaded)
    network.set_train(False)

    f = open(save_file_path, 'w')
    iterator = dataset.create_tuple_iterator()
    for item in iterator:
        output = network(*item[:-1])
        for i in output:
            f.write(str(i[1]) + '\n')
            f.flush()
    f.close()

if __name__ == '__main__':
    # !/bin/bash
    candidate_file = "data/candidate.test.txt"
    score_file = "output/score.txt"
    predict_file = "output/predict.txt"
    load_checkpoint_path = "save_model2/match_kn_gene-3_568.ckpt"

    run_duconv()

    extract_predict_utterance(candidate_file,score_file,predict_file)


    # step 6: if the original file has answers, you can run the following command to get result
    # if the original file not has answers, you can upload the ./output/test.result.final
    # to the website(https://ai.baidu.com/broad/submission?dataset=duconv) to get the official automatic evaluation
    # python
    # src / eval.py ${predict_file} > predict.log