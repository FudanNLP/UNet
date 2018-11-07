#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : dataset.py
# Author            : Sun Fu <cstsunfu@gmail.com>
# Date              : 01.07.2018
# Last Modified Date: 05.11.2018
# Last Modified By  : Sun Fu <cstsunfu@gmail.com>
import ujson as json
import numpy as np
import torch
import random

from allennlp.modules.elmo import batch_to_ids
random.seed(2023)


def get_data(filename) :
    with open(filename, 'r', encoding='utf-8') as f :
        data = json.load(f)
    return data


def load_data(opts):
    print('load data...')
    data_path = opts['data_path']

    train_data = get_data(data_path + 'train.json')
    dev_data = get_data(data_path + 'dev.json')
    word2id = get_data(data_path + 'word2id.json')
    id2word = {v : k for k, v in word2id.items()}
    char2id = get_data(data_path + 'char2id.json')
    pos2id = get_data(data_path + 'pos2id.json')
    ner2id = get_data(data_path + 'ner2id.json')

    opts['char_size'] = int(np.max(list(char2id.values())) + 1)
    opts['pos_size'] = int(np.max(list(pos2id.values())) + 1)
    opts['ner_size'] = int(np.max(list(ner2id.values())) + 1)

    return train_data, dev_data, word2id, id2word, char2id, opts


def get_batches(data, batch_size, evaluation=False) :

    if not evaluation:
        indices = list(range(len(data['context_ids'])))
        random.shuffle(indices)
        for key in data.keys():
            if isinstance(data[key], int):
                continue
            data[key] = [data[key][i] for i in indices]

    for i in range(0, len(data['context_ids']), batch_size) :
        batch_size = len(data['context_ids'][i:i+batch_size])
        yield (data['context_ids'][i:i+batch_size],
               data['context_char_ids'][i:i+batch_size],
               data['context_pos_ids'][i:i+batch_size],
               data['context_ner_ids'][i:i+batch_size],
               data['context_match_origin'][i:i+batch_size],
               data['context_match_lower'][i:i+batch_size],
               data['context_match_lemma'][i:i+batch_size],
               data['context_tf'][i:i+batch_size],
               data['ques_ids'][i:i+batch_size],
               data['ques_char_ids'][i:i+batch_size],
               data['ques_pos_ids'][i:i+batch_size],
               data['ques_ner_ids'][i:i+batch_size],
               data['y1'][i:i+batch_size],
               data['y2'][i:i+batch_size],
               data['id'][i:i+batch_size],
               batch_to_ids(data['context_tokens'][i:i+batch_size]),
               batch_to_ids(data['ques_tokens'][i:i+batch_size]),
               data['has_ans'][i:i+batch_size],
               data['ques_tf'][i:i+batch_size],
               data['ques_match_origin'][i:i+batch_size],
               data['ques_match_lower'][i:i+batch_size],
               data['ques_match_lemma'][i:i+batch_size],
               data['y1p'][i:i+batch_size],
               data['y2p'][i:i+batch_size])
