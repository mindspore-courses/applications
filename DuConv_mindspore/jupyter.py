# # 下载数据集到本地
# import requests
# def download(url):
#     req = requests.get(url)
#     filename = url.split('/')[-1]
#     if req.status_code != 200:
#         print('下载异常')
#         return
#     try:
#         with open(filename, 'wb') as f:
#             # req.content为获取html的内容
#             f.write(req.content)
#             print('下载成功')
#     except Exception as e:
#         print(e)
#
#
# if __name__ == '__main__':
#     url = 'https://dataset-bj.cdn.bcebos.com/duconv/train.txt.gz'  #
#     download(url)
#     ur2 = 'https://dataset-bj.cdn.bcebos.com/duconv/dev.txt.gz'  #
#     download(ur2)
#     ur3 = 'https://dataset-bj.cdn.bcebos.com/duconv/test_1.txt.gz'  #
#     download(ur3)
#
# #移动文件位置到data
# import os
# from pathlib import Path
# import shutil
# if not os.path.exists('data'):
#     os.makedirs('data')
# os.rename('test_1.txt.gz', 'test.txt.gz')
# shutil.move('./train.txt.gz', './data')
# shutil.move('./dev.txt.gz', './data')
# shutil.move('./test.txt.gz', './data')
#
# import gzip
# def un_gz(file_name):
#     # 获取文件的名称，去掉后缀名
#     f_name = file_name.replace(".gz", "")
#     # 开始解压
#     g_file = gzip.GzipFile(file_name)
#     # 读取解压后的文件，并写入去掉后缀名的同名文件（即得到解压后的文件）
#     open(f_name, "wb+").write(g_file.read())
#     g_file.close()
#
# #将gz解压为txt
# un_gz('./data/train.txt.gz')
# un_gz('./data/dev.txt.gz')
# un_gz('./data/test.txt.gz')
#
# # 将下载下来的三个数据集截断为较小的数据集
# # train 100行
# # test 20行
# # dev 10行
# import os
#
#
# def slice_data(file_name,length):
#     with open(file_name, encoding='utf-8') as fic:
#         lines = [fic.readline() for _ in range(length)]
#         fic.close()
#
#     with open(file_name, 'w', encoding='utf-8') as fic:
#         fic.writelines(lines)
#         fic.close()
#     return 0
#
# slice_data('./data/train.txt',100)
# slice_data('./data/test.txt',20)
# slice_data('./data/dev.txt',10)
#
# # 数据预处理
# # build_candidate_set_from_corpus
#
# # from __future__ import print_function
# import sys
# import json
# import random
# import collections
#
# # reload(sys)
# # sys.setdefaultencoding('utf8')
# import functools
#
#
# def cmp(a, b):
#     len_a, len_b = len(a[1]), len(b[1])
#     if len_a > len_b:
#         return 1
#     elif len_a < len_b:
#         return -1
#     else:
#         return 0
#
# # 从语料库文件中生成候选集
# def build_candidate_set_from_corpus(corpus_file, candidate_set_file):
#     """
#     build candidate set from corpus
#     """
#     candidate_set_gener = {}
#     candidate_set_mater = {}
#     candidate_set_list = []
#     slot_dict = {"topic_a": 1, "topic_b": 1}
#     with open(corpus_file, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             conversation = json.loads(line.strip(), encoding="utf-8",
#                                       object_pairs_hook=collections.OrderedDict)
#
#             chat_path = conversation["goal"]
#             knowledge = conversation["knowledge"]
#             session = conversation["conversation"]
#
#             topic_a = chat_path[0][1]
#             topic_b = chat_path[0][2]
#             domain_a = None
#             domain_b = None
#             cover_att_list = [[["topic_a", topic_a],
#                                ["topic_b", topic_b]]] * len(session)
#             for j, [s, p, o] in enumerate(knowledge):
#                 p_key = ""
#                 if topic_a.replace(' ', '') == s.replace(' ', ''):
#                     p_key = "topic_a_" + p.replace(' ', '')
#                     if u"领域" == p:
#                         domain_a = o
#                 elif topic_b.replace(' ', '') == s.replace(' ', ''):
#                     p_key = "topic_b_" + p.replace(' ', '')
#                     if u"领域" == p:
#                         domain_b = o
#
#                 for k, utterance in enumerate(session):
#                     if k % 2 == 1:
#                         continue
#                     if o in utterance and o != topic_a and o != topic_b and p_key != "":
#                         cover_att_list[k].append([p_key, o])
#
#                 slot_dict[p_key] = 1
#
#             assert domain_a is not None and domain_b is not None
#
#             for j, utterance in enumerate(session):
#                 if j % 2 == 1:
#                     continue
#                 key = '_'.join([domain_a, domain_b, str(j)])
#
#                 cover_att = sorted(
#                     cover_att_list[j], key=functools.cmp_to_key(cmp), reverse=True)
#
#                 utterance_gener = utterance
#                 for [p_key, o] in cover_att:
#                     utterance_gener = utterance_gener.replace(o, p_key)
#
#                 if "topic_a_topic_a_" not in utterance_gener and \
#                    "topic_a_topic_b_" not in utterance_gener and \
#                    "topic_b_topic_a_" not in utterance_gener and \
#                    "topic_b_topic_b_" not in utterance_gener:
#                     if key in candidate_set_gener:
#                         candidate_set_gener[key].append(utterance_gener)
#                     else:
#                         candidate_set_gener[key] = [utterance_gener]
#
#                 utterance_mater = utterance
#                 for [p_key, o] in [["topic_a", topic_a], ["topic_b", topic_b]]:
#                     utterance_mater = utterance_mater.replace(o, p_key)
#
#                 if key in candidate_set_mater:
#                     candidate_set_mater[key].append(utterance_mater)
#                 else:
#                     candidate_set_mater[key] = [utterance_mater]
#
#                 candidate_set_list.append(utterance_mater)
#
#     fout = open(candidate_set_file, 'w')
#     fout.write(json.dumps(candidate_set_gener, ensure_ascii=False) + "\n")
#     fout.write(json.dumps(candidate_set_mater, ensure_ascii=False) + "\n")
#     fout.write(json.dumps(candidate_set_list, ensure_ascii=False) + "\n")
#     fout.write(json.dumps(slot_dict, ensure_ascii=False))
#     fout.close()
#
# import sys
# import json
# import collections
#
# # 采样
# def convert_session_to_sample(session_file, sample_file):
#     """
#     convert_session_to_sample
#     """
#     fout = open(sample_file, 'w', encoding='utf-8')
#     with open(session_file, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             session = json.loads(line.strip(), encoding="utf-8",
#                                  object_pairs_hook=collections.OrderedDict)
#             conversation = session["conversation"]
#
#             for j in range(0, len(conversation), 2):
#                 sample = collections.OrderedDict()
#                 sample["goal"] = session["goal"]
#                 sample["knowledge"] = session["knowledge"]
#                 sample["history"] = conversation[:j]
#                 sample["response"] = conversation[j]
#
#                 sample = json.dumps(sample, ensure_ascii=False)
#
#                 fout.write(sample + "\n")
#
#     fout.close()
#
# # from __future__ import print_function
# import sys
# import json
# import random
# import collections
#
# # 加载候选集，返回列表
# def load_candidate_set(candidate_set_file):
#     """
#     load candidate set
#     """
#     candidate_set = []
#     for line in open(candidate_set_file):
#         candidate_set.append(json.loads(line.strip(), encoding="utf-8"))
#
#     return candidate_set
#
# # 选择合法的候选
# def candidate_slection(candidate_set, knowledge_dict, slot_dict, candidate_num=10):
#     """
#     candidate slection
#     """
#     random.shuffle(candidate_set)
#     candidate_legal = []
#     for candidate in candidate_set:
#         is_legal = True
#         for slot in slot_dict:
#             if slot in ["topic_a", "topic_b"]:
#                 continue
#             if slot in candidate:
#                 if slot not in knowledge_dict:
#                     is_legal = False
#                     break
#                 w_ = random.choice(knowledge_dict[slot])
#                 candidate = candidate.replace(slot, w_)
#
#         for slot in ["topic_a", "topic_b"]:
#             if slot in candidate:
#                 if slot not in knowledge_dict:
#                     is_legal = False
#                     break
#                 w_ = random.choice(knowledge_dict[slot])
#                 candidate = candidate.replace(slot, w_)
#
#         if is_legal and candidate not in candidate_legal:
#             candidate_legal.append(candidate)
#
#         if len(candidate_legal) >= candidate_num:
#             break
#
#     return candidate_legal
#
# # 选择用于对话系统的候选集
# def get_candidate_for_conversation(conversation, candidate_set, candidate_num=10):
#     """
#     get candidate for conversation
#     """
#     candidate_set_gener, candidate_set_mater, candidate_set_list, slot_dict = candidate_set
#
#     chat_path = conversation["goal"]
#     knowledge = conversation["knowledge"]
#     history = conversation["history"]
#
#     topic_a = chat_path[0][1]
#     topic_b = chat_path[0][2]
#     domain_a = None
#     domain_b = None
#     knowledge_dict = {"topic_a": [topic_a], "topic_b": [topic_b]}
#     for i, [s, p, o] in enumerate(knowledge):
#         p_key = ""
#         if topic_a.replace(' ', '') == s.replace(' ', ''):
#             p_key = "topic_a_" + p.replace(' ', '')
#             if u"领域" == p:
#                 domain_a = o
#         elif topic_b.replace(' ', '') == s.replace(' ', ''):
#             p_key = "topic_b_" + p.replace(' ', '')
#             if u"领域" == p:
#                 domain_b = o
#
#         if p_key == "":
#             continue
#
#         if p_key in knowledge_dict:
#             knowledge_dict[p_key].append(o)
#         else:
#             knowledge_dict[p_key] = [o]
#
#     assert domain_a is not None and domain_b is not None
#
#     key = '_'.join([domain_a, domain_b, str(len(history))])
#
#     candidate_legal = []
#     if key in candidate_set_gener:
#         candidate_legal.extend(candidate_slection(candidate_set_gener[key],
#                                                   knowledge_dict, slot_dict,
#                                                   candidate_num=candidate_num - len(candidate_legal)))
#
#     if len(candidate_legal) < candidate_num and key in candidate_set_mater:
#         candidate_legal.extend(candidate_slection(candidate_set_mater[key],
#                                                   knowledge_dict, slot_dict,
#                                                   candidate_num=candidate_num - len(candidate_legal)))
#
#     if len(candidate_legal) < candidate_num:
#         candidate_legal.extend(candidate_slection(candidate_set_list,
#                                                   knowledge_dict, slot_dict,
#                                                   candidate_num=candidate_num - len(candidate_legal)))
#
#     return candidate_legal
#
#
# # 为语料库构建候选集
# def construct_candidate_for_corpus(corpus_file, candidate_set_file, candidate_file, candidate_num=10):
#     """
#     construct candidate for corpus
#
#     case of data in corpus_file:
#     {
#         "goal": [["START", "休 · 劳瑞", "蕾切儿 · 哈伍德"]],
#         "knowledge": [["休 · 劳瑞", "评论", "完美 的 男人"]],
#         "history": ["你 对 明星 有没有 到 迷恋 的 程度 呢 ？",
#                     "一般 吧 ， 毕竟 年纪 不 小 了 ， 只是 追星 而已 。"]
#     }
#
#     case of data in candidate_file:
#     {
#         "goal": [["START", "休 · 劳瑞", "蕾切儿 · 哈伍德"]],
#         "knowledge": [["休 · 劳瑞", "评论", "完美 的 男人"]],
#         "history": ["你 对 明星 有没有 到 迷恋 的 程度 呢 ？",
#                     "一般 吧 ， 毕竟 年纪 不 小 了 ， 只是 追星 而已 。"],
#         "candidate": ["我 说 的 是 休 · 劳瑞 。",
#                       "我 说 的 是 休 · 劳瑞 。"]
#     }
#     """
#     candidate_set = load_candidate_set(candidate_set_file)
#     fout_text = open(candidate_file, 'w', encoding="utf-8")
#     with open(corpus_file, 'r', encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             conversation = json.loads(line.strip(), encoding="utf-8",
#                                       object_pairs_hook=collections.OrderedDict)
#             candidates = get_candidate_for_conversation(conversation,
#                                                         candidate_set,
#                                                         candidate_num=candidate_num)
#             conversation["candidate"] = candidates
#
#             conversation = json.dumps(conversation, ensure_ascii=False)
#             fout_text.write(conversation + "\n")
#
#     fout_text.close()
#
# """
# File: convert_conversation_corpus_to_model_text.py
# """
# # 将对话语料库转换为符合模型输入的文本形式
# import sys
# import re
# import json
# import collections
# # from construct_candidate import get_candidate_for_conversation
#
#
# def parser_char_for_word(word):
#     """
#     parser char for word
#     """
#     if word.isdigit():
#         return word
#     for i in range(len(word)):
#         if word[i] >= u'\u4e00' and word[i] <= u'\u9fa5':
#             word_out = " ".join([t for t in word])
#             word_out = re.sub(" +", " ", word_out)
#             return word_out
#     return word
#
#
# def parser_char_for_text(text):
#     """
#     parser char for text
#     """
#     words = text.strip().split()
#     for i, word in enumerate(words):
#         words[i] = parser_char_for_word(word)
#     return re.sub(" +", " ", ' '.join(words))
#
#
# def topic_generalization_for_text(text, topic_list):
#     """
#     topic generalization for text
#     """
#     for key, value in topic_list:
#         text = text.replace(value, key)
#
#     return text
#
#
# def topic_generalization_for_list(text_list, topic_list):
#     """
#     topic generalization for list
#     """
#     for i, text in enumerate(text_list):
#         text_list[i] = topic_generalization_for_text(text, topic_list)
#
#     return text_list
#
#
# def preprocessing_for_one_conversation(text,
#                                        candidate_num=10,
#                                        use_knowledge=True,
#                                        topic_generalization=False,
#                                        for_predict=True):
#     """
#     preprocessing for one conversation
#     """
#
#     conversation = json.loads(text.strip(), encoding="utf-8",
#                               object_pairs_hook=collections.OrderedDict)
#
#     goal = conversation["goal"]
#     knowledge = conversation["knowledge"]
#     history = conversation["history"]
#     if not for_predict:
#         response = conversation["response"]
#
#     topic_a = goal[0][1]
#     topic_b = goal[0][2]
#     for i, [s, p, o] in enumerate(knowledge):
#         if u"领域" == p:
#             if topic_a == s:
#                 domain_a = o
#             elif topic_b == s:
#                 domain_b = o
#
#     topic_dict = {}
#     if u"电影" == domain_a:
#         topic_dict["video_topic_a"] = topic_a
#     else:
#         topic_dict["person_topic_a"] = topic_a
#
#     if u"电影" == domain_b:
#         topic_dict["video_topic_b"] = topic_b
#     else:
#         topic_dict["person_topic_b"] = topic_b
#
#     if "candidate" in conversation:
#         candidates = conversation["candidate"]
#     else:
#         assert candidate_num > 0
#         candidates = get_candidate_for_conversation(conversation,
#                                                     candidate_num=candidate_num)
#
#     if topic_generalization:
#         topic_list = sorted(topic_dict.items(),
#                             key=lambda item: len(item[1]), reverse=True)
#
#         goal = [topic_generalization_for_list(spo, topic_list) for spo in goal]
#
#         knowledge = [topic_generalization_for_list(
#             spo, topic_list) for spo in knowledge]
#
#         history = [topic_generalization_for_text(utterance, topic_list)
#                    for utterance in history]
#
#         for i, candidate in enumerate(candidates):
#             candidates[i] = topic_generalization_for_text(
#                 candidate, topic_list)
#
#         if not for_predict:
#             response = topic_generalization_for_text(response, topic_list)
#
#     goal = ' [PATH_SEP] '.join([parser_char_for_text(' '.join(spo))
#                                 for spo in goal])
#     knowledge = ' [KN_SEP] '.join([parser_char_for_text(' '.join(spo))
#                                    for spo in knowledge])
#     history = ' [INNER_SEP] '.join([parser_char_for_text(utterance)
#                                     for utterance in history]) \
#         if len(history) > 0 else '[START]'
#
#     model_text = []
#
#     for candidate in candidates:
#         candidate = parser_char_for_text(candidate)
#         if use_knowledge:
#             text_ = '\t'.join(["0", history, candidate, goal, knowledge])
#         else:
#             text_ = '\t'.join(["0", history, candidate])
#
#         text_ = re.sub(" +", " ", text_)
#         model_text.append(text_)
#
#     if not for_predict:
#         candidates.append(response)
#         response = parser_char_for_text(response)
#         if use_knowledge:
#             text_ = '\t'.join(["1", history, response, goal, knowledge])
#         else:
#             text_ = '\t'.join(["1", history, response])
#
#         text_ = re.sub(" +", " ", text_)
#         model_text.append(text_)
#
#     return model_text, candidates
#
#
# def convert_conversation_corpus_to_model_text(corpus_file, text_file,
#                                               use_knowledge=True,
#                                               topic_generalization=False,
#                                               for_predict=True):
#     """
#     convert conversation corpus to model text
#     """
#     fout_text = open(text_file, 'w', encoding='utf-8')
#     with open(corpus_file, 'r',encoding='utf-8') as f:
#         for i, line in enumerate(f):
#             model_text, _ = preprocessing_for_one_conversation(
#                 line.strip(), candidate_num=0,
#                 use_knowledge=use_knowledge,
#                 topic_generalization=topic_generalization,
#                 for_predict=for_predict)
#
#             for text in model_text:
#                 fout_text.write(text + "\n")
#
#     fout_text.close()
#
# """
# File: build_dict.py
# """
# # 根据准备的文本数据，构建字符级字典
#
# import sys
#
#
# def build_dict(corpus_file, dict_file):
#     """
#     build words dict
#     """
#     dict = {}
#     max_frequency = 1
# #     for line in open(corpus_file, 'r', encoding='utf-8'):
#     with open(corpus_file,'r',encoding='utf-8') as f:
#         for line in f:
#             conversation = line.strip().split('\t')
#             for i in range(1, len(conversation), 1):
#                 words = conversation[i].split(' ')
#                 for word in words:
#                     if word in dict:
#                         dict[word] = dict[word] + 1
#                         if dict[word] > max_frequency:
#                             max_frequency = dict[word]
#                     else:
#                         dict[word] = 1
#
#         dict["[PAD]"] = max_frequency + 4
#         dict["[UNK]"] = max_frequency + 3
#         dict["[CLS]"] = max_frequency + 2
#         dict["[SEP]"] = max_frequency + 1
#
#         words = sorted(dict.items(), key=lambda item: item[1], reverse=True)
#
#         fout = open(dict_file, 'w', encoding='utf-8')
#         for word, frequency in words:
#             fout.write(word + '\n')
#
#         fout.close()
#
# # 数据预处理
# DICT_NAME = "data/gene.dict"
# USE_KNOWLEDGE = 1
# TOPIC_GENERALIZATION = 1
#
# FOR_PREDICT = 0
# CANDIDATE_NUM = 9
# # data目录下的tarin.txt,dev.txt,test.txt都已作截断处理，长度分别为100行，10行和20行
# DATA_TYPE = ['train', 'dev', 'test']
# INPUT_PATH = 'data/'
# candidate_set_file = 'data/candidate_set.txt'
# for i in range(len(DATA_TYPE)):
#     corpus_file = 'data/{DATA}.txt'.format(DATA=DATA_TYPE[i])
#     sample_file = 'data/sample.{DATA}.txt'.format(DATA=DATA_TYPE[i])
#     candidate_file = 'data/candidate.{DATA}.txt'.format(DATA=DATA_TYPE[i])
#     text_file = 'data/build.{DATA}.txt'.format(DATA=DATA_TYPE[i])
#
#     # step 1: build candidate set from session data for negative training cases and predicting candidates
#     if DATA_TYPE[i] == 'train':
#         build_candidate_set_from_corpus(corpus_file, candidate_set_file)
# #         build_dict(text_file, DICT_NAME)
#
#     # step 2: firstly have to convert session data to sample data
#     if DATA_TYPE[i] == 'test':
#         sample_file = corpus_file
#         FOR_PREDICT = 1
#         CANDIDATE_NUM = 10
#     else:
#         convert_session_to_sample(corpus_file, sample_file)
#
#     # step 3: construct candidate for sample data
#     construct_candidate_for_corpus(sample_file, candidate_set_file,
#                                    candidate_file, CANDIDATE_NUM)
#
#     # step 4: convert sample data with candidates to text data required by the model
#     convert_conversation_corpus_to_model_text(
#         candidate_file, text_file, USE_KNOWLEDGE, TOPIC_GENERALIZATION, FOR_PREDICT)
#
#     # step 5: build dict from the training data, here we build character dict for model
#     if DATA_TYPE[i] == "train":
#         build_dict(text_file, DICT_NAME)
#
# # corpus_file='datasets/train.txt'
# # candidate_set_file='datasets/candidate_set.txt'
# # build_candidate_set_from_corpus(corpus_file,candidate_set_file)
#
#
# # sample_file='datasets/test.txt'
# # FOR_PREDICT=1
# # CANDIDATE_NUM=10
#
# import argparse
# from typing import Sequence
# import numpy as np
# from mindspore.mindrecord import FileWriter
# from mindspore.log import logging
#
#
# # 利用mindspore将txt文本数据转换格式为.mindrecord数据
#
# def load_dict(vocab_path):
#     """
#     load vocabulary dict
#     """
#     vocab_dict = {}
#     idx = 0
#     for line in open(vocab_path, encoding='utf-8'):
#         line = line.strip()
#         vocab_dict[line] = idx
#         idx += 1
#     return vocab_dict
#
#
# class DataProcessor:
#     def __init__(self, task_name, vocab_path, max_seq_len, do_lower_case):
#         self.task_name = task_name
#         self.max_seq_len = max_seq_len
#         self.do_lower_case = do_lower_case
#         self.vocab_dict = load_dict(vocab_path)
#
#     def get_labels(self):
#         return ["0", "1"]
#
#     def _read_data(self, input_file):
#         lines = []
#         with open(input_file, 'r', encoding='utf-8') as f:
#             for line in f:
#                 line = line.rstrip('\n').split('\t')
#                 lines.append(line)
#         return lines
#
#     def _create_examples(self, input_file):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         lines = self._read_data(input_file)
#         for line in lines:
#             context_text = line[1]
#             label_text = line[0]
#             response_text = line[2]
#             if 'kn' in self.task_name:
#                 kn_text = "%s [SEP] %s" % (line[3], line[4])
#             else:
#                 kn_text = None
#             examples.append(
#                 InputExample(context_text=context_text, response_text=response_text, \
#                              kn_text=kn_text, label_text=label_text))
#         return examples
#
#     def _convert_example_to_record(self, example, labels, max_seq_len, vocab_dict):
#         """Converts a single `InputExample` into a single `InputFeatures`."""
#         feature = convert_single_example(example, labels, max_seq_len, vocab_dict)
#         return feature
#
#     def file_based_convert_examples_to_features(self, input_file, output_file):
#         """"Convert a set of `InputExample`s to a MindDataset file."""
#         examples = self._create_examples(input_file)
#
#         writer = FileWriter(file_name=output_file, shard_num=1)
#         nlp_schema = {
#             "context_id": {"type": "int64", "shape": [-1]},
#             "context_segment_id": {"type": "int64", "shape": [-1]},
#             "context_pos_id": {"type": "int64", "shape": [-1]},
#             "labels_list": {"type": "int64", "shape": [-1]}
#         }
#         if 'kn' in self.task_name:
#             nlp_schema['kn_id'] = {"type": "int64", "shape": [-1]}
#             nlp_schema['kn_seq_length'] = {"type": "int64", "shape": [-1]}
#
#         writer.add_schema(nlp_schema, "proprocessed dataset")
#         data = []
#         for index, example in enumerate(examples):
#             if index % 10000 == 0:
#                 logging.info("Writing example %d of %d" % (index, len(examples)))
#             record = self._convert_example_to_record(example, self.get_labels(), self.max_seq_len, self.vocab_dict)
#             sample = {
#                 "context_id": np.array(record.context_ids, dtype=np.int64),
#                 "context_pos_id": np.array(record.context_pos_ids, dtype=np.int64),
#                 "context_segment_id": np.array(record.segment_ids, dtype=np.int64),
#                 "labels_list": np.array([record.label_ids], dtype=np.int64),
#             }
#             if 'kn' in self.task_name:
#                 sample['kn_id'] = np.array(record.kn_ids, dtype=np.int64)
#                 sample['kn_seq_length'] = np.array(record.kn_seq_length, dtype=np.int64)
#
#             data.append(sample)
#         writer.write_raw_data(data)
#         writer.commit()
#
#
# class InputExample(object):
#     """A single training/test example"""
#
#     def __init__(self, context_text, response_text, kn_text, label_text):
#         self.context_text = context_text
#         self.response_text = response_text
#         self.kn_text = kn_text
#         self.label_text = label_text
#
#
# class InputFeatures(object):
#     """input features datas"""
#
#     def __init__(self, context_ids, context_pos_ids, segment_ids, kn_ids, kn_seq_length, label_ids):
#         self.context_ids = context_ids
#         self.context_pos_ids = context_pos_ids
#         self.segment_ids = segment_ids
#         self.kn_ids = kn_ids
#         self.kn_seq_length = kn_seq_length
#         self.label_ids = label_ids
#
#
# def convert_tokens_to_ids(tokens, vocab_dict):
#     """
#     convert input ids
#     """
#     ids = []
#     for token in tokens:
#         if token in vocab_dict:
#             ids.append(vocab_dict[token])
#         else:
#             ids.append(vocab_dict['[UNK]'])
#     return ids
#
#
# def convert_single_example(example, label_list, max_seq_length, vocab_dict):
#     """Converts a single `InputExample` into a single `InputFeatures`."""
#     label_map = {}
#     for (i, label) in enumerate(label_list):
#         label_map[label] = i
#     if example.context_text:
#         tokens_context = example.context_text
#         tokens_context = tokens_context.split()
#     else:
#         tokens_context = []
#
#     if example.response_text:
#         tokens_response = example.response_text
#         tokens_response = tokens_response.split()
#     else:
#         tokens_response = []
#
#     if example.kn_text:
#         tokens_kn = example.kn_text
#         tokens_kn = tokens_kn.split()
#         tokens_kn = tokens_kn[0: min(len(tokens_kn), max_seq_length)]
#     else:
#         tokens_kn = []
#
#     tokens_response = tokens_response[0: min(50, len(tokens_response))]
#     if len(tokens_context) > max_seq_length - len(tokens_response) - 3:
#         tokens_context = tokens_context[len(tokens_context) \
#                                         + len(tokens_response) - max_seq_length + 3:]
#
#     context_tokens = []
#     segment_ids = []
#
#     context_tokens.append("[CLS]")
#     segment_ids.append(0)
#     context_tokens.extend(tokens_context)
#     segment_ids.extend([0] * len(tokens_context))
#     context_tokens.append("[SEP]")
#     segment_ids.append(0)
#
#     context_tokens.extend(tokens_response)
#     segment_ids.extend([1] * len(tokens_response))
#     context_tokens.append("[SEP]")
#     segment_ids.append(1)
#
#     context_ids = convert_tokens_to_ids(context_tokens, vocab_dict)
#     context_ids = context_ids + [0] * (max_seq_length - len(context_ids))
#     context_pos_ids = list(range(len(context_ids))) + [0] * (max_seq_length - len(context_ids))
#     segment_ids = segment_ids + [0] * (max_seq_length - len(segment_ids))
#     label_ids = label_map[example.label_text]
#     if tokens_kn:
#         kn_ids = convert_tokens_to_ids(tokens_kn, vocab_dict)
#         kn_ids = kn_ids[0: min(max_seq_length, len(kn_ids))]
#         kn_seq_length = len(kn_ids)
#         kn_ids = kn_ids + [0] * (max_seq_length - kn_seq_length)
#     else:
#         kn_ids = []
#         kn_seq_length = 0
#
#     # print(len(context_ids), len(context_pos_ids), len(segment_ids), len(kn_ids), kn_seq_length, label_ids)
#     feature = InputFeatures(
#         context_ids=context_ids,
#         context_pos_ids=context_pos_ids,
#         segment_ids=segment_ids,
#         kn_ids=kn_ids,
#         kn_seq_length=kn_seq_length,
#         label_ids=label_ids)
#
#     return feature
#
#
# def reader_main(task_name, vocab_path, max_seq_len, input_file, output_file, do_lower_case=''):
#     #     parser = argparse.ArgumentParser(description="read dataset and save it to minddata")
#     #     parser.add_argument("--task_name", type=str, default="match", choices=["match", "match_kn", "match_kn_gene"], help="vocab file")
#     #     parser.add_argument("--vocab_path", type=str, default="", help="vocab file")
#     #     parser.add_argument("--max_seq_len", type=int, default=256,
#     #                         help="The maximum total input sequence length after WordPiece tokenization. "
#     #                         "Sequences longer than this will be truncated, and sequences shorter "
#     #                         "than this will be padded.")
#     #     parser.add_argument("--do_lower_case", type=str, default="true",
#     #                         help="Whether to lower case the input text. "
#     #                         "Should be True for uncased models and False for cased models.")
#
#     #     parser.add_argument("--input_file", type=str, default="", help="raw data file")
#     #     parser.add_argument("--output_file", type=str, default="", help="minddata file")
#     #     args_opt = parser.parse_args()
#
#     processer = DataProcessor(task_name, vocab_path, max_seq_len, True if do_lower_case == "true" else False)
#     processer.file_based_convert_examples_to_features(input_file, output_file)
#
# # convert_dataset.sh
# # 需要reader.py
# # TASK_NAME=match_kn_gene
# # DICT_NAME=data/gene.dict
# reader_main(task_name='match_kn_gene', vocab_path='data/gene.dict', max_seq_len=256,
#             input_file='data/build.train.txt', output_file='data/train.mindrecord')
# reader_main(task_name='match_kn_gene', vocab_path='data/gene.dict', max_seq_len=256,
#             input_file='data/build.dev.txt', output_file='data/dev.mindrecord')
# reader_main(task_name='match_kn_gene', vocab_path='data/gene.dict', max_seq_len=256,
#             input_file='data/build.test.txt', output_file='data/test.mindrecord')
#
# # python src/reader.py - -task_name =${TASK_NAME} \
# #                      - -max_seq_len = 256 \
# #                      - -vocab_path =${DICT_NAME} \
# #                      - -input_file = data/build.train.txt \
# #                      - -output_file = data/train.mindrecord
# # python src/reader.py - -task_name =${TASK_NAME} \
# #                      - -max_seq_len = 256 \
# #                      - -vocab_path =${DICT_NAME} \
# #                      - -input_file = data/build.dev.txt \
# #                      - -output_file = data/dev.mindrecord
# # python src/reader.py - -task_name =${TASK_NAME} \
# #                      - -max_seq_len = 256 \
# #                      - -vocab_path =${DICT_NAME} \
# #                      - -input_file = data/build.test.txt \
# #                      - -output_file = data/test.mindrecord
#
#
#
#
#
#
#

"""
File: extract_predict_utterance.py
"""

import sys
import json
import collections

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
from mindspore import log as logger

######bert.py
######
import mindspore
import mindspore.nn as nn
import mindspore.ops as P
import mindspore.numpy as mnp
import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
from mindspore import Tensor
from mindspore.common.initializer import Zero, TruncatedNormal, Constant

######rnn_encoder.py
#######
########
import mindspore.nn as nn
from mindspore.nn.layer import activation
import mindspore.ops as P

######model.py
########
#######
import mindspore.nn as nn
import mindspore.ops as P
# from .bert import BertModel
# from .rnn_encoder import RNNEncoder
from mindspore.common.initializer import TruncatedNormal
import mindspore.common.dtype as mstype

########lr_schsdule
########
########

import mindspore.ops as P
import mindspore.common.dtype as mstype
from mindspore.nn.learning_rate_schedule import LearningRateSchedule
from mindspore import Tensor


#########callbacks.py
#########
#########

import math
import time
from mindspore.train.callback import Callback
from numpy.lib.function_base import average

#saydgfusdhfihs
def create_dataset(batch_size, device_num=1, rank=0, do_shuffle=True, data_file_path=None, use_knowledge=False):
    """create train dataset"""
    if use_knowledge:
        colums_list = ["context_id", "context_segment_id", "context_pos_id", "kn_id", "kn_seq_length", "labels_list"]
    else:
        colums_list = ["context_id", "context_segment_id", "context_pos_id", "labels_list"]

    # apply repeat operations
    data_set = ds.MindDataset(data_file_path,
                              columns_list=colums_list,
                              shuffle=do_shuffle,
                              num_shards=device_num,
                              shard_id=rank
                              )

    ori_dataset_size = data_set.get_dataset_size()
    print('origin dataset size: ', ori_dataset_size)
    type_cast_op = C.TypeCast(mstype.int32)
    data_set = data_set.map(operations=type_cast_op, input_columns="context_id")
    data_set = data_set.map(operations=type_cast_op, input_columns="context_pos_id")
    data_set = data_set.map(operations=type_cast_op, input_columns="context_segment_id")
    data_set = data_set.map(operations=type_cast_op, input_columns="labels_list")
    if use_knowledge:
        data_set = data_set.map(operations=type_cast_op, input_columns="kn_id")
        data_set = data_set.map(operations=type_cast_op, input_columns="kn_seq_length")
    # apply batch operations
    data_set = data_set.batch(batch_size, drop_remainder=True)
    logger.info("data size: {}".format(data_set.get_dataset_size()))
    logger.info("repeat count: {}".format(data_set.get_repeat_count()))
    return data_set


class GELU(nn.Cell):
    def __init__(self):
        super().__init__()
        self.erf = P.Erf()
        self.sqrt = P.Sqrt()
        self.const0 = Tensor(0.5, mindspore.float32)
        self.const1 = Tensor(1.0, mindspore.float32)
        self.const2 = Tensor(2.0, mindspore.float32)

    def construct(self, x):
        return x * self.const0 * (self.const1 + self.erf(x / self.sqrt(self.const2)))


class MaskedFill(nn.Cell):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.select = P.Select()
        self.fill = P.Fill()
        self.cast = P.Cast()

    def construct(self, inputs: Tensor, mask: Tensor):
        mask = self.cast(mask, mstype.bool_)
        masked_value = self.fill(mindspore.float32, inputs.shape, self.value)
        output = self.select(mask, masked_value, inputs)
        return output


class ScaledDotProductAttention(nn.Cell):
    def __init__(self, d_k, dropout):
        super().__init__()
        self.scale = Tensor(d_k, mindspore.float32)
        self.matmul = nn.MatMul()
        self.transpose = P.Transpose()
        self.softmax = nn.Softmax(axis=-1)
        self.sqrt = P.Sqrt()
        self.masked_fill = MaskedFill(-1e9)

        if dropout > 0.0:
            self.dropout = nn.Dropout(1 - dropout)
        else:
            self.dropout = None

    def construct(self, Q, K, V, attn_mask):
        K = self.transpose(K, (0, 1, 3, 2))
        scores = self.matmul(Q, K) / self.sqrt(
            self.scale)  # scores : [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        scores = self.masked_fill(scores, attn_mask)  # Fills elements of self tensor with value where mask is one.
        # scores = scores + attn_mask
        attn = self.softmax(scores)
        context = self.matmul(attn, V)
        if self.dropout is not None:
            context = self.dropout(context)
        return context, attn


class MultiHeadAttention(nn.Cell):
    def __init__(self, d_model, n_heads, dropout, initializer_range=0.02):
        super().__init__()
        self.n_heads = n_heads
        self.W_Q = nn.Dense(d_model, d_model)
        self.W_K = nn.Dense(d_model, d_model)
        self.W_V = nn.Dense(d_model, d_model)
        self.linear = nn.Dense(d_model, d_model)
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model, "embed_dim must be divisible by num_heads"
        self.layer_norm = nn.LayerNorm((d_model,), epsilon=1e-6)
        self.attention = ScaledDotProductAttention(self.head_dim, dropout)
        # ops
        self.transpose = P.Transpose()
        self.expanddims = P.ExpandDims()
        self.tile = P.Tile()

    def construct(self, Q, K, V, attn_mask):
        # q: [batch_size x len_q x d_model], k: [batch_size x len_k x d_model], v: [batch_size x len_k x d_model]
        residual, batch_size = Q, Q.shape[0]
        q_s = self.W_Q(Q).view((batch_size, -1, self.n_heads, self.head_dim))
        k_s = self.W_K(K).view((batch_size, -1, self.n_heads, self.head_dim))
        v_s = self.W_V(V).view((batch_size, -1, self.n_heads, self.head_dim))
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)
        q_s = self.transpose(q_s, (0, 2, 1, 3))  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.transpose(k_s, (0, 2, 1, 3))  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.transpose(v_s, (0, 2, 1, 3))  # v_s: [batch_size x n_heads x len_k x d_v]

        attn_mask = self.expanddims(attn_mask, 1)
        attn_mask = self.tile(attn_mask, (1, self.n_heads, 1, 1))  # attn_mask : [batch_size x n_heads x len_q x len_k]

        # context: [batch_size x n_heads x len_q x d_v], attn: [batch_size x n_heads x len_q(=len_k) x len_k(=len_q)]
        context, attn = self.attention(q_s, k_s, v_s, attn_mask)
        context = self.transpose(context, (0, 2, 1, 3)).view(
            (batch_size, -1, self.n_heads * self.head_dim))  # context: [batch_size x len_q x n_heads * d_v]
        output = self.linear(context)
        return self.layer_norm(output + residual), attn  # output: [batch_size x len_q x d_model]


activation_map = {
    'relu': nn.ReLU(),
    'gelu': GELU(),
}


def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.shape
    batch_size, len_k = seq_k.shape

    # pad_attn_mask = P.ExpandDims()(P.ZerosLike()(seq_k), 1)
    pad_attn_mask = P.ExpandDims()(P.Equal()(seq_k, 0), 1)
    pad_attn_mask = P.Cast()(pad_attn_mask, mstype.int32)
    pad_attn_mask = P.BroadcastTo((batch_size, len_q, len_k))(pad_attn_mask)
    # pad_attn_mask = P.Cast()(pad_attn_mask, mstype.bool_)
    return pad_attn_mask


class BertConfig:
    def __init__(self,
                 seq_length=256,
                 vocab_size=32000,
                 hidden_size=256,
                 num_hidden_layers=4,
                 num_attention_heads=8,
                 intermediate_size=1024,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=256,
                 type_vocab_size=2,
                 initializer_range=0.02):
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range


class PoswiseFeedForwardNet(nn.Cell):
    def __init__(self, d_model, d_ff, activation: str = 'gelu', initializer_range=0.02, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Dense(d_model, d_ff)
        self.fc2 = nn.Dense(d_ff, d_model)
        self.activation = activation_map.get(activation, nn.GELU())
        self.layer_norm = nn.LayerNorm((d_model,), epsilon=1e-6)
        self.dropout = nn.Dropout(1 - dropout)

    def construct(self, inputs):
        residual = inputs
        outputs = self.fc1(inputs)
        outputs = self.activation(outputs)

        outputs = self.fc2(outputs)
        outputs = self.dropout(outputs)
        return self.layer_norm(outputs + residual)


class BertEmbeddings(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.tok_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.pos_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.seg_embed = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm((config.hidden_size,), epsilon=1e-6)
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)

        self.expand_dims = P.ExpandDims()

    def construct(self, x, seg, pos=None):
        seq_len = x.shape[1]
        if pos is None:
            pos = mnp.arange(seq_len)
            pos = P.BroadcastTo(x.shape)(self.expand_dims(pos, 0))
        seg_embedding = self.seg_embed(seg)
        tok_embedding = self.tok_embed(x)
        embedding = tok_embedding + self.pos_embed(pos) + seg_embedding
        embedding = self.norm(embedding)
        embedding = self.dropout(embedding)
        return embedding


class BertEncoderLayer(nn.Cell):
    def __init__(self, d_model, n_heads, d_ff, activation, attention_dropout, dropout):
        super().__init__()
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads, attention_dropout)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff, activation, dropout)

    def construct(self, enc_inputs, enc_self_attn_mask):
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs, attn


class BertEncoder(nn.Cell):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.CellList([
            BertEncoderLayer(config.hidden_size,
                             config.num_attention_heads,
                             config.intermediate_size,
                             config.hidden_act,
                             config.attention_probs_dropout_prob,
                             config.hidden_dropout_prob)
            for _ in range(config.num_hidden_layers)
        ])

    def construct(self, inputs, enc_self_attn_mask):
        outputs = inputs
        for layer in self.layers:
            outputs, enc_self_attn = layer(outputs, enc_self_attn_mask)
        return outputs


class BertModel(nn.Cell):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = nn.Dense(config.hidden_size, config.hidden_size, activation='tanh')

    def construct(self, input_ids, segment_ids, position_ids=None):
        outputs = self.embeddings(input_ids, segment_ids, position_ids)
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)
        outputs = self.encoder(outputs, enc_self_attn_mask)
        h_pooled = self.pooler(outputs[:, 0])
        return outputs, h_pooled



class RNNEncoder(nn.Cell):
    def __init__(self, input_size, hidden_size, bidirectional, num_layers, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super().__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions

        self.bidirectional = bidirectional
        self.embeddings = embeddings
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self.total_hidden_dim = hidden_size * num_layers
            self.bridge = nn.Dense(self.total_hidden_dim, self.total_hidden_dim, activation='relu')

    def construct(self, inputs, seq_length):
        emb = self.embeddings(inputs)
        memory_bank, encoder_final = self.rnn(emb, seq_length=seq_length)

        if self.use_bridge:
            shape = encoder_final.shape
            encoder_final = self.bridge(encoder_final.view(-1, self.total_hidden_dim)).view(shape)

        if self.bidirectional:
            batch_size = encoder_final.shape[1]
            encoder_final = encoder_final.view(self.num_layers, 2, batch_size, self.hidden_size) \
                .swapaxes(1, 2).view(self.num_layers, batch_size, self.hidden_size * 2)

        return encoder_final, memory_bank


class Attention(nn.Cell):
    def __init__(self, memory_size, hidden_size):
        super().__init__()
        self.bmm = P.BatchMatMul()
        self.concat = P.Concat(-1)
        self.softmax = nn.Softmax()
        self.linear_project = nn.SequentialCell([
            nn.Dense(hidden_size + memory_size, hidden_size),
            nn.Tanh()
        ])

    def construct(self, query, memory):
        attn = self.bmm(query, memory.swapaxes(1, 2))
        weights = self.softmax(attn)
        weighted_memory = self.bmm(weights, memory)
        project_output = self.linear_project(self.concat((weighted_memory, query)))
        return project_output


class Retrieval(nn.Cell):
    def __init__(self, config, use_kn=False):
        super().__init__()
        self.bert = BertModel(config)
        self.use_kn = use_kn
        if self.use_kn:
            self.kn_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
            self.memory = RNNEncoder(
                config.hidden_size,
                config.hidden_size,
                True,
                1,
                config.hidden_dropout_prob,
                self.kn_embeddings,
                True
            )
            self.attention = Attention(config.hidden_size, config.hidden_size)
        self.fc = nn.Dense(config.hidden_size, 2, weight_init=TruncatedNormal(config.initializer_range))
        self.dropout = nn.Dropout(1 - config.hidden_dropout_prob)

    def construct(self, input_ids, segment_ids, position_ids=None, kn_ids=None, seq_length=None):
        # print(kn_ids)
        if len(seq_length.shape) != 1:
            seq_length = P.Squeeze(1)(seq_length)
        _, h_pooled = self.bert(input_ids, segment_ids, position_ids)
        h_pooled = P.ExpandDims()(h_pooled, 1)
        if self.use_kn:
            _, memory_bank = self.memory(kn_ids, seq_length)
            cls_feats = self.attention(h_pooled, memory_bank)
        else:
            cls_feats = h_pooled
        cls_feats = self.dropout(cls_feats.squeeze(1))
        logits = self.fc(cls_feats)
        # print(cls_feats.shape)
        return logits


class RetrievalWithLoss(nn.Cell):
    def __init__(self, config, use_kn):
        super().__init__()
        self.network = Retrieval(config, use_kn)
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.squeeze = P.Squeeze(1)

    def construct(self, *inputs):
        # print(inputs[-1].shape)
        out = self.network(*inputs[:-1])
        # print(out.shape, inputs[-1].shape)
        labels = self.squeeze(inputs[-1])
        return self.loss(out, labels)


class RetrievalWithSoftmax(nn.Cell):
    def __init__(self, config, use_kn):
        super().__init__()
        self.network = Retrieval(config, use_kn)
        self.softmax = nn.Softmax()

    def construct(self, *inputs):
        out = self.network(*inputs)
        out = self.softmax(out)
        return out


class Noam(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps, learning_rate=1.0):
        super().__init__()
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.learning_rate = learning_rate
        self.pow = P.Pow()
        self.min = P.Minimum()
        self.cast = P.Cast()
        self.const0 = Tensor(-0.5, mstype.float32)
        self.const1 = Tensor(-1.5, mstype.float32)

    def construct(self, global_step):
        p = self.cast(self.min(
            self.pow(global_step, self.const0),
            self.pow(self.warmup_steps, self.const1) * global_step),
            mstype.float32)
        return self.learning_rate * self.pow(self.d_model, self.const0) * p



class TimeMonitor(Callback):
    """
    Monitor the time in training.

    Args:
        data_size (int): How many steps are the intervals between print information each time.
            if the program get `batch_num` during training, `data_size` will be set to `batch_num`,
            otherwise `data_size` will be used. Default: None.

    Raises:
        ValueError: If data_size is not positive int.
    """

    def __init__(self, per_print_times=1):
        super(TimeMonitor, self).__init__()
        self._per_print_times = per_print_times
        self.epoch_time = time.time()
        self.time_list = []

    def step_begin(self, run_context):
        self.epoch_time = time.time()

    def step_end(self, run_context):
        step_seconds = (time.time() - self.epoch_time) * 1000
        self.time_list.append(step_seconds)
        cb_params = run_context.original_args()
        if self._per_print_times != 0 and cb_params.cur_step_num % self._per_print_times == 0:
            print("per step time: {:5.3f} ms".format(average(self.time_list)), flush=True)
        self.time_list = []

class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    Note:
        if per_print_times is 0 do not print loss.
    Args:
        per_print_times (int): Print loss every times. Default: 1.
    """
    def __init__(self, dataset_size=-1):
        super(LossCallBack, self).__init__()
        self._dataset_size = dataset_size
    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            print("epoch: {}, current epoch percent: {}, step: {}, outputs are {}"
                  .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, str(cb_params.net_outputs)),
                  flush=True)
        else:
            print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                               str(cb_params.net_outputs)), flush=True)



import mindspore
import argparse
import numpy as np
from mindspore import context
# from src.model import RetrievalWithLoss
# from src.bert import BertConfig
# from src.lr_schedule import Noam
# from src.callbacks import TimeMonitor
from mindspore.nn.optim import Adam
from mindspore.train.model import Model
from mindspore.nn import TrainOneStepCell
# from src.dataset import create_dataset
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, LossMonitor

def run_duconv1():
    parser = argparse.ArgumentParser(description='train duconv')

    epoch = 10
    task_name = 'match_kn'
    max_seq_length = 512
    batch_size = 8096
    vocab_size = 14373
    learning_rate = 0.1
    weight_decay = 0.01
    warmup_proportion = 0.1
    train_data_file_path = ""
    train_data_shuffle = "true"
    save_checkpoint_path = ""

    epoch = 3

    max_seq_length = 256
    batch_size = 8
    train_data_file_path = "./data/train.mindrecord"
    save_checkpoint_path = "save_model2/"

    """run duconv task"""

    context.set_context(mode=context.GRAPH_MODE, max_call_depth=10000)
    use_kn = "match_kn"

    config = BertConfig(seq_length=max_seq_length, vocab_size=vocab_size)
    dataset = create_dataset(batch_size, data_file_path=train_data_file_path,
                             do_shuffle=(train_data_shuffle.lower() == "true"), use_knowledge=use_kn)
    steps_per_epoch = dataset.get_dataset_size()
    print(steps_per_epoch)

    max_train_steps = epoch * steps_per_epoch
    warmup_steps = int(max_train_steps * warmup_proportion)
    keep_checkpoint = int(max_train_steps / 1000) + 1
    network = RetrievalWithLoss(config, use_kn)
    lr_schedule = Noam(config.hidden_size, warmup_steps, learning_rate)
    optimizer = Adam(network.trainable_params(), lr_schedule)
    network_one_step = TrainOneStepCell(network, optimizer)
    model = Model(network_one_step)
    ckpt_config = CheckpointConfig(save_checkpoint_steps=1000, keep_checkpoint_max=keep_checkpoint)
    ckpoint_cb = ModelCheckpoint(prefix=task_name,
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    callbacks = [TimeMonitor(100), LossMonitor(100), ckpoint_cb]

    model.train(epoch, dataset, callbacks, dataset_sink_mode=False)

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
# from src.model import RetrievalWithSoftmax
# from src.bert import BertConfig
# from src.dataset import create_dataset

# from __future__ import print_function
import sys
import json
import random
import collections

# reload(sys)
# sys.setdefaultencoding('utf8')
import functools


def cmp(a, b):
    len_a, len_b = len(a[1]), len(b[1])
    if len_a > len_b:
        return 1
    elif len_a < len_b:
        return -1
    else:
        return 0


# def build_candidate_set_from_corpus(corpus_file, candidate_set_file):
#     """
#     build candidate set from corpus
#     """
#     candidate_set_gener = {}
#     candidate_set_mater = {}
#     candidate_set_list = []
#     slot_dict = {"topic_a": 1, "topic_b": 1}
#     with open(corpus_file, 'r', encoding="utf-8") as f:
#         for i, line in enumerate(f):
#             conversation = json.loads(line.strip(), encoding="utf-8", object_pairs_hook=collections.OrderedDict)
#
#             chat_path = conversation["goal"]
#             knowledge = conversation["knowledge"]
#             session = conversation["conversation"]
#
#             topic_a = chat_path[0][1]
#             topic_b = chat_path[0][2]
#             domain_a = None
#             domain_b = None
#             cover_att_list = [[["topic_a", topic_a], ["topic_b", topic_b]]] * len(session)
#             for j, [s, p, o] in enumerate(knowledge):
#                 p_key = ""
#                 if topic_a.replace(' ', '') == s.replace(' ', ''):
#                     p_key = "topic_a_" + p.replace(' ', '')
#                     if u"领域" == p:
#                         domain_a = o
#                 elif topic_b.replace(' ', '') == s.replace(' ', ''):
#                     p_key = "topic_b_" + p.replace(' ', '')
#                     if u"领域" == p:
#                         domain_b = o
#
#                 for k, utterance in enumerate(session):
#                     if k % 2 == 1: continue
#                     if o in utterance and o != topic_a and o != topic_b and p_key != "":
#                         cover_att_list[k].append([p_key, o])
#
#                 slot_dict[p_key] = 1
#
#             assert domain_a is not None and domain_b is not None
#
#             for j, utterance in enumerate(session):
#                 if j % 2 == 1: continue
#                 key = '_'.join([domain_a, domain_b, str(j)])
#
#                 cover_att = sorted(cover_att_list[j], key=functools.cmp_to_key(cmp), reverse=True)
#
#                 utterance_gener = utterance
#                 for [p_key, o] in cover_att:
#                     utterance_gener = utterance_gener.replace(o, p_key)
#
#                 if "topic_a_topic_a_" not in utterance_gener and \
#                         "topic_a_topic_b_" not in utterance_gener and \
#                         "topic_b_topic_a_" not in utterance_gener and \
#                         "topic_b_topic_b_" not in utterance_gener:
#                     if key in candidate_set_gener:
#                         candidate_set_gener[key].append(utterance_gener)
#                     else:
#                         candidate_set_gener[key] = [utterance_gener]
#
#                 utterance_mater = utterance
#                 for [p_key, o] in [["topic_a", topic_a], ["topic_b", topic_b]]:
#                     utterance_mater = utterance_mater.replace(o, p_key)
#
#                 if key in candidate_set_mater:
#                     candidate_set_mater[key].append(utterance_mater)
#                 else:
#                     candidate_set_mater[key] = [utterance_mater]
#
#                 candidate_set_list.append(utterance_mater)
#
#     fout = open(candidate_set_file, 'w', encoding="utf-8")
#     fout.write(json.dumps(candidate_set_gener, ensure_ascii=False) + "\n")
#     fout.write(json.dumps(candidate_set_mater, ensure_ascii=False) + "\n")
#     fout.write(json.dumps(candidate_set_list, ensure_ascii=False) + "\n")
#     fout.write(json.dumps(slot_dict, ensure_ascii=False))
#     fout.close()
#
#
# def main():
#     """
#     main
#     """
#     build_candidate_set_from_corpus(sys.argv[1], sys.argv[2])


#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################



def extract_predict_utterance(sample_file, score_file, output_file):
    """
    convert_result_for_eval
    """
    sample_list = [line.strip() for line in open(sample_file, 'r')]
    score_list = [line.strip() for line in open(score_file, 'r')]

    fout = open(output_file, 'w')
    index = 0
    for i, sample in enumerate(sample_list):
        sample = json.loads(sample, encoding="utf-8", object_pairs_hook=collections.OrderedDict)

        candidates = sample["candidate"]
        scores = score_list[index: index + len(candidates)]

        pridict = candidates[0]
        max_score = float(scores[0])
        for j, score in enumerate(scores):
            score = float(score)
            if score > max_score:
                pridict = candidates[j]
                max_score = score

        if "response" in sample:
            response = sample["response"]
            fout.write(pridict + "\t" + response + "\n")
        else:
            fout.write(pridict + "\n")

        index = index + len(candidates)

    fout.close()

######
#eval.py
######
"""
File: eval.py
"""
def eval(predict_file):
    import sys
    import math
    from collections import Counter

    # if len(sys.argv) < 2:
    #     print("Usage: " + sys.argv[0] + " eval_file")
    #     print("eval file format: pred_response \t gold_response")
    #     exit()

    def get_dict(tokens, ngram, gdict=None):
        """
        get_dict
        """
        token_dict = {}
        if gdict is not None:
            token_dict = gdict
        tlen = len(tokens)
        for i in range(0, tlen - ngram + 1):
            ngram_token = "".join(tokens[i:(i + ngram)])
            if token_dict.get(ngram_token) is not None:
                token_dict[ngram_token] += 1
            else:
                token_dict[ngram_token] = 1
        return token_dict

    def count(pred_tokens, gold_tokens, ngram, result):
        """
        count
        """
        cover_count, total_count = result
        pred_dict = get_dict(pred_tokens, ngram)
        gold_dict = get_dict(gold_tokens, ngram)
        cur_cover_count = 0
        cur_total_count = 0
        for token, freq in pred_dict.items():
            if gold_dict.get(token) is not None:
                gold_freq = gold_dict[token]
                cur_cover_count += min(freq, gold_freq)
            cur_total_count += freq
        result[0] += cur_cover_count
        result[1] += cur_total_count

    def calc_bp(pair_list):
        """
        calc_bp
        """
        c_count = 0.0
        r_count = 0.0
        for pair in pair_list:
            pred_tokens, gold_tokens = pair
            c_count += len(pred_tokens)
            r_count += len(gold_tokens)
        bp = 1
        if c_count < r_count:
            bp = math.exp(1 - r_count / c_count)
        return bp

    def calc_cover_rate(pair_list, ngram):
        """
        calc_cover_rate
        """
        result = [0.0, 0.0]  # [cover_count, total_count]
        for pair in pair_list:
            pred_tokens, gold_tokens = pair
            count(pred_tokens, gold_tokens, ngram, result)
        cover_rate = result[0] / result[1]
        return cover_rate

    def calc_bleu(pair_list):
        """
        calc_bleu
        """
        bp = calc_bp(pair_list)
        cover_rate1 = calc_cover_rate(pair_list, 1)
        cover_rate2 = calc_cover_rate(pair_list, 2)
        cover_rate3 = calc_cover_rate(pair_list, 3)
        bleu1 = 0
        bleu2 = 0
        bleu3 = 0
        if cover_rate1 > 0:
            bleu1 = bp * math.exp(math.log(cover_rate1))
        if cover_rate2 > 0:
            bleu2 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2)) / 2)
        if cover_rate3 > 0:
            bleu3 = bp * math.exp((math.log(cover_rate1) + math.log(cover_rate2) + math.log(cover_rate3)) / 3)
        return [bleu1, bleu2]

    def calc_distinct_ngram(pair_list, ngram):
        """
        calc_distinct_ngram
        """
        ngram_total = 0.0
        ngram_distinct_count = 0.0
        pred_dict = {}
        for predict_tokens, _ in pair_list:
            get_dict(predict_tokens, ngram, pred_dict)
        for key, freq in pred_dict.items():
            ngram_total += freq
            ngram_distinct_count += 1
            # if freq == 1:
            #    ngram_distinct_count += freq
        return ngram_distinct_count / ngram_total

    def calc_distinct(pair_list):
        """
        calc_distinct
        """
        distinct1 = calc_distinct_ngram(pair_list, 1)
        distinct2 = calc_distinct_ngram(pair_list, 2)
        return [distinct1, distinct2]

    def calc_f1(data):
        """
        calc_f1
        """
        golden_char_total = 0.0
        pred_char_total = 0.0
        hit_char_total = 0.0
        for response, golden_response in data:
            golden_response = "".join(golden_response)
            response = "".join(response)
            common = Counter(response) & Counter(golden_response)
            hit_char_total += sum(common.values())
            golden_char_total += len(golden_response)
            pred_char_total += len(response)
        p = hit_char_total / pred_char_total
        r = hit_char_total / golden_char_total
        f1 = 2 * p * r / (p + r)
        return f1

    # eval_file = sys.argv[1]
    eval_file = predict_file
    sents = []
    for line in open(eval_file):
        tk = line.strip().split("\t")
        if len(tk) < 2:
            continue
        pred_tokens = tk[0].strip().split(" ")
        gold_tokens = tk[1].strip().split(" ")
        sents.append([pred_tokens, gold_tokens])
    # calc f1
    f1 = calc_f1(sents)
    # calc bleu
    bleu1, bleu2 = calc_bleu(sents)
    # calc distinct
    distinct1, distinct2 = calc_distinct(sents)

    output_str = "F1: %.2f%%\n" % (f1 * 100)
    output_str += "BLEU1: %.3f%%\n" % bleu1
    output_str += "BLEU2: %.3f%%\n" % bleu2
    output_str += "DISTINCT1: %.3f%%\n" % distinct1
    output_str += "DISTINCT2: %.3f%%\n" % distinct2
    sys.stdout.write(output_str)

    def main():
        """
        main
        """
        extract_predict_utterance(sys.argv[1],
                                  sys.argv[2],
                                  sys.argv[3])




def run_duconv2():
    """run duconv task"""
    parser = argparse.ArgumentParser(description='train duconv')
    candidate_file = "data/candidate.test.txt"
    score_file = "output/score.txt"
    predict_file = "output/predict.txt"
    load_checkpoint_path = "save_model2/match_kn-3_568.ckpt"
    task_name = "match_kn"
    max_seq_length = 128
    batch_size = 1
    eval_data_file_path = "data/test.mindrecord"
    load_checkpoint_path = "save_model2/match_kn-3_568.ckpt"
    save_file_path = "output/score.txt"
    vocab_size = 14373

    """run duconv task"""
    # args = parse_args()
    context.set_context(mode=context.GRAPH_MODE, max_call_depth=10000)
    use_kn = "match_kn"
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
    extract_predict_utterance(candidate_file, score_file, predict_file)

if __name__ == '__main__':

    run_duconv1()
    run_duconv2()

