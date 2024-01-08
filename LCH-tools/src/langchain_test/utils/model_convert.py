import collections
import os
import json
import shutil
import argparse
import paddle.fluid.dygraph as D
from paddle import fluid
from mindspore import Tensor
from mindspore.train.serialization import save_checkpoint

def build_params_map(attention_num=12):
    """
    build params map from paddle-paddle's ERNIE to transformer's ernie
    :return:
    """
    weight_map = collections.OrderedDict({
        'word_embedding': "tok_embeddings.ernie_embedding_lookup.embedding_table",
        'pos_embedding': "tok_embeddings.ernie_embedding_postprocessor.full_position_embedding.embedding_table",
        'sent_embedding': "tok_embeddings.ernie_embedding_postprocessor.token_type_embedding.embedding_table",
        'pre_encoder_layer_norm_scale': 'tok_embeddings.ernie_embedding_postprocessor.layernorm.gamma',
        'pre_encoder_layer_norm_bias': 'tok_embeddings.ernie_embedding_postprocessor.layernorm.beta',
    })
    # add attention layers
    for i in range(attention_num):
        weight_map[f'encoder_layer_{i}_multi_head_att_query_fc.w_0'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.attention.attention.query_layer.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_query_fc.b_0'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.attention.attention.query_layer.bias'
        weight_map[f'encoder_layer_{i}_multi_head_att_key_fc.w_0'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.attention.attention.key_layer.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_key_fc.b_0'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.attention.attention.key_layer.bias'
        weight_map[f'encoder_layer_{i}_multi_head_att_value_fc.w_0'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.attention.attention.value_layer.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_value_fc.b_0'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.attention.attention.value_layer.bias'
        weight_map[f'encoder_layer_{i}_multi_head_att_output_fc.w_0'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.attention.output.dense.weight'
        weight_map[f'encoder_layer_{i}_multi_head_att_output_fc.b_0'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.attention.output.dense.bias'
        weight_map[f'encoder_layer_{i}_post_att_layer_norm_scale'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.attention.output.layernorm.gamma'
        weight_map[f'encoder_layer_{i}_post_att_layer_norm_bias'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.attention.output.layernorm.beta'
        weight_map[f'encoder_layer_{i}_ffn_fc_0.w_0'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.intermediate.weight'
        weight_map[f'encoder_layer_{i}_ffn_fc_0.b_0'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.intermediate.bias'
        weight_map[f'encoder_layer_{i}_ffn_fc_1.w_0'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.output.dense.weight'
        weight_map[f'encoder_layer_{i}_ffn_fc_1.b_0'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.output.dense.bias'
        weight_map[f'encoder_layer_{i}_post_ffn_layer_norm_scale'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.output.layernorm.gamma'
        weight_map[f'encoder_layer_{i}_post_ffn_layer_norm_bias'] = \
            f'tok_embeddings.ernie_encoder.layers.{i}.output.layernorm.beta'
    # add pooler
    weight_map.update(
        {
            'pooled_fc.w_0': 'tok_embeddings.dense.weight',
            'pooled_fc.b_0': 'tok_embeddings.dense.bias',
            'cls_out_w': 'ernie.dense_1.weight',
            'cls_out_b': 'ernie.dense_1.bias'
        }
    )
    return weight_map

def extract_and_convert(input_dir, output_dir):
    """extract weights and convert to mindspore"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config = json.load(open(os.path.join(input_dir, 'ernie_config.json'), 'rt', encoding='utf-8'))
    print('=' * 21 + 'save vocab file' + '=' * 20)
    shutil.copyfile(os.path.join(input_dir, 'vocab.txt'), os.path.join(output_dir, 'vocab.txt'))
    print('=' * 20 + 'extract weights' + '=' * 20)
    state_dict = []
    weight_map = build_params_map(attention_num=config['num_hidden_layers'])
    with fluid.dygraph.guard():
        paddle_paddle_params, _ = D.load_dygraph(os.path.join(input_dir, 'params'))
    for weight_name, weight_value in paddle_paddle_params.items():
        if weight_name not in weight_map.keys():
            continue
        #print(weight_name, weight_value.shape)
        if 'w_0' in weight_name \
            or 'post_att_layer_norm_scale' in weight_name \
            or 'post_ffn_layer_norm_scale' in weight_name \
            or 'cls_out_w' in weight_name:
            weight_value = weight_value.transpose()
        state_dict.append({'name': weight_map[weight_name], 'data': Tensor(weight_value)})
        print(weight_name, '->', weight_map[weight_name], weight_value.shape)
    save_checkpoint(state_dict, os.path.join(output_dir, "ernie.ckpt"))

def run_convert():
    """run convert"""
    parser = argparse.ArgumentParser(description="run convert")
    parser.add_argument("--input_dir", type=str, default="", help="Pretrained model dir")
    parser.add_argument("--output_dir", type=str, default="", help="Converted model dir")
    args_opt = parser.parse_args()
    extract_and_convert(args_opt.input_dir, args_opt.output_dir)

if __name__ == '__main__':
    run_convert()