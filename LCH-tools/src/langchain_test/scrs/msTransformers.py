import json
import logging
import os
import shutil
import stat
from collections import OrderedDict
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional
import requests
import numpy as np
from numpy import ndarray
import transformers
from mindspore import context
from transformers import BertTokenizer
import mindspore as ms
from mindspore import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from tqdm.autonotebook import trange
import math
import queue
import tempfile
from distutils.dir_util import copy_tree
from scrs import ErnieModel, ErnieConfig
from mindformers import BertTokenizer

logger = logging.getLogger(__name__)
tokenizer_path = "/home/ma-user/work/mindformers/ernie_model/tokenizer.model"


class Embeddings(nn.Cell):
    def __init__(self,
                 vocab_size=32000,
                 hidden_size=768,
                 pooling_mode: str = None,
                 pooling_mode_cls_token: bool = False,
                 pooling_mode_max_tokens: bool = False,
                 pooling_mode_mean_tokens: bool = True,
                 pooling_mode_mean_sqrt_len_tokens: bool = False,
                 ):
        super().__init__()
        ernie_net_cfg = ErnieConfig(
            seq_length=512,
            vocab_size=18000,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="relu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=513,
            type_vocab_size=2,
            initializer_range=0.02,
            use_relative_positions=False,
            dtype=ms.float32,
            compute_type=ms.float16,
        )

        self.tok_embeddings = ErnieModel(ernie_net_cfg, False)
        print("model initing")
        self.config_keys = ['word_embedding_dimension', 'pooling_mode_cls_token', 'pooling_mode_mean_tokens',
                            'pooling_mode_max_tokens', 'pooling_mode_mean_sqrt_len_tokens']

        if pooling_mode is not None:  # Set pooling mode by string
            pooling_mode = pooling_mode.lower()
            assert pooling_mode in ['mean', 'max', 'cls']
            pooling_mode_cls_token = (pooling_mode == 'cls')
            pooling_mode_max_tokens = (pooling_mode == 'max')
            pooling_mode_mean_tokens = (pooling_mode == 'mean')

        self.word_embedding_dimension = hidden_size
        self.pooling_mode_cls_token = pooling_mode_cls_token
        self.pooling_mode_mean_tokens = pooling_mode_mean_tokens
        self.pooling_mode_max_tokens = pooling_mode_max_tokens
        self.pooling_mode_mean_sqrt_len_tokens = pooling_mode_mean_sqrt_len_tokens

        pooling_mode_multiplier = sum([pooling_mode_cls_token, pooling_mode_max_tokens, pooling_mode_mean_tokens,
                                       pooling_mode_mean_sqrt_len_tokens])
        self.pooling_output_dimension = (pooling_mode_multiplier * hidden_size)

    def construct(self, features, mask):
        embeddings, _, _ = self.tok_embeddings(features, mask)
        ## Pooling strategy
        output_vectors = []
        if self.pooling_mode_cls_token:
            cls_token = features.get('cls_token_embeddings', embeddings[:, 0])  # Take first token by default
            output_vectors.append(cls_token)
        if self.pooling_mode_max_tokens:
            input_mask_expanded = mask.unsqueeze(-1)
            input_mask_expanded = input_mask_expanded.expand_as(ms.Tensor(embeddings.shape, dtype=ms.int64))
            embeddings[input_mask_expanded == 0] = -1e9  # Set padding tokens to large negative value
            max_over_time = ms.ops.max(embeddings, 1)[0]
            output_vectors.append(max_over_time)
        if self.pooling_mode_mean_tokens or self.pooling_mode_mean_sqrt_len_tokens:
            input_mask_expanded = mask.unsqueeze(-1)
            input_mask_expanded = ms.ops.broadcast_to(input_mask_expanded, embeddings.shape)

            sum_embeddings = ms.ops.sum(embeddings * input_mask_expanded, 1)

            sum_mask = ms.ops.sum(input_mask_expanded, 1)
            # mins = ms.Tensor(shape=sum_mask, init=ms.common.initializer.One(), dtype=ms.float16)*1e-9
            mins = ms.ops.full_like(sum_mask, fill_value=1e-9)
            sum_mask = ms.ops.clamp(sum_mask, min=mins)

            if self.pooling_mode_mean_tokens:
                output_vectors.append(sum_embeddings / sum_mask)
            if self.pooling_mode_mean_sqrt_len_tokens:
                output_vectors.append(sum_embeddings / ms.ops.sqrt(sum_mask))

        output_vector = ms.ops.cat(output_vectors, 1)
        # features.update({'sentence_embedding': output_vector})
        return output_vector, embeddings


class SentenceTransformer():
    """
    Loads or create a SentenceTransformer model, that can be used to map sentences / text to embeddings.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :param modules: This parameter can be used to create custom SentenceTransformer models from scratch.
    :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
    :param cache_folder: Path to store models
    :param use_auth_token: HuggingFace authentication token to download private models.
    """

    def __init__(self, model_name_or_path: Optional[str] = None,
                 model_tokenizer = None,
                 modules: Optional[Iterable[nn.Cell]] = None,
                 device: Optional[str] = None,
                 cache_folder: Optional[str] = None,
                 use_auth_token: Union[bool, str, None] = None
                 ):
        # super(SentenceTransformer, self).__init__()
        self._model_card_vars = {}
        self._model_card_text = None
        self._model_config = {}

        # self.token = Tokenizer(model_path=tokenizer_path)
        self.token = BertTokenizer(model_tokenizer)

        self.modules = Embeddings()

        param_dict = ms.load_checkpoint(model_name_or_path)
        ms.load_param_into_net(self.modules, param_dict)
        self.modules.set_train(False)

        self.model = ms.Model(self.modules)

        '''if modules is not None and not isinstance(modules, OrderedDict):
            modules = OrderedDict([(str(idx), module) for idx, module in enumerate(modules)])
        '''

    def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
        """
        Computes sentence embeddings

        :param sentences: the sentences to embed
        :param batch_size: the batch size used for the computation
        :param show_progress_bar: Output a progress bar when encode sentences
        :param output_value:  Default sentence_embedding, to get sentence embeddings. Can be set to token_embeddings to get wordpiece token embeddings. Set to None, to get all output values
        :param convert_to_numpy: If true, the output is a list of numpy vectors. Else, it is a list of pytorch tensors.
        :param convert_to_tensor: If true, you get one large tensor as return. Overwrites any setting from convert_to_numpy
        :param device: Which torch.device to use for the computation
        :param normalize_embeddings: If set to true, returned vectors will have length 1. In that case, the faster dot-product (util.dot_score) instead of cosine similarity can be used.

        :return:
           By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned. If convert_to_numpy, a numpy matrix is returned.
        """

        # self.modules[0].auto_model.set_train(False)
        if show_progress_bar is None:
            show_progress_bar = (
                    logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)

        if convert_to_tensor:
            convert_to_numpy = False

        if output_value != 'sentence_embedding':
            convert_to_tensor = False
            convert_to_numpy = False

        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences,
                                                     '__len__'):  # Cast an individual sentence to a list with length 1
            sentences = [sentences]
            input_was_string = True

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index:start_index + batch_size]

            features = [self.token.encode(sentence) for sentence in sentences_batch]
            '''lenss = 0
            for tmp in features:
                lenss = max(lenss, len(tmp))'''

            total_len = 512

            ## 优化性能
            features = np.array(
                [np.concatenate(
                    [x, np.ones(total_len - len(x), dtype=int) * self.token.pad_token_id], axis=0)
                 if len(x) < total_len else x[:total_len] for x in features], np.int32)


            # features = np.array(
            #     [ms.ops.concat(
            #         (ms.Tensor(x), ms.ops.ones(total_len - len(x), dtype=ms.int64) * self.token.pad_token_id),
            #         0) if len(x) < total_len else ms.Tensor(x[:total_len]) for x in features], np.int64)

            '''total_len = 512
            features = [self.token.encode(sentence, bos=True, eos=False) for sentence in sentences_batch]
            tok = np.array(
                [ms.ops.concat((ms.Tensor(x), ms.ops.ones(total_len - len(x), dtype=ms.int64)*self.token.pad_token_id),
                            0) if len(x) < total_len else ms.Tensor(x) for x in features], np.int64)

            input_text_mask = np.array(tok != self.token.pad_token_id, np.int64)'''
            # features = self.token.encode(sentences_batch, padding="max_length")
            input_text_mask = np.array(features != self.token.pad_token_id, np.int64)

            out_features, embeds = self.model.predict(ms.Tensor(features, dtype=ms.int32),
                                                      ms.Tensor(input_text_mask, dtype=ms.int32))
            '''if output_value == 'token_embeddings':
                embeddings = []
                for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                    last_mask_id = len(attention) - 1
                    while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                        last_mask_id -= 1

                    embeddings.append(token_emb[0:last_mask_id + 1])
            elif output_value is None:  # Return all outputs
                embeddings = []
                for sent_idx in range(len(out_features['sentence_embedding'])):
                    row = {name: out_features[name][sent_idx] for name in out_features}
                    embeddings.append(row)
            else:  # Sentence embeddings
                embeddings = out_features[output_value]

                if normalize_embeddings:
                    embeddings = ms.ops.L2Normalize(1)(embeddings)'''

            # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
            # embeddings = ms.ops.L2Normalize(1)(out_features)

            all_embeddings.extend(out_features)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if convert_to_tensor:
            all_embeddings = ms.ops.stack(all_embeddings)
        elif convert_to_numpy:
            all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def _text_length(self, text: Union[List[int], List[List[int]]]):
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, '__len__'):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def tokenize(self, texts: Union[List[str], List[Dict], List[Tuple[str, str]]]):
        """
        Tokenizes the texts
        """
        return self.modules[0].tokenize(texts)

    def predict(self, input):

        return self.modules(input)
