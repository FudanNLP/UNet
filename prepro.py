#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : prepro.py
# Author            : Sun Fu <cstsunfu@gmail.com>
# Date              : 23.06.2018
# Last Modified Date: 07.11.2018
# Last Modified By  : Sun Fu <cstsunfu@gmail.com>
import random
import ujson as json
import numpy as np
import pickle as pkl
import spacy
import re
import os

from tqdm import tqdm
from collections import Counter

# random.seed(1023)
nlp = spacy.load("en", parser=False)


def space_extend(matchobj):
    return ' ' + matchobj.group(0) + ' '


def pre_proc(text) :
    text = re.sub(u'-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2015|%|\[|\]|:|\(|\)|/', space_extend, text)
    # text = text.strip(' \n')
    text = re.sub('\s+', ' ', text)
    return text


def word_tokenize(sent):
    doc = nlp(sent)

    text = []
    tag = []
    ent = []
    lemma = []
    for token in doc :
        text.append(token.text)
        tag.append(token.tag_)
        ent.append(token.ent_type_)
        lemma.append(token.lemma_)

    return text, tag, ent, lemma


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter, pos_counter, ner_counter, ques_word_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = ''.join(("NoAnswer ", para["context"].replace("''", '" ').replace("``", '" ')))

                raw_context = context

                ### additional preproc ###

                context = pre_proc(context)
                context_tokens, context_tags, context_ents, context_lemmas = word_tokenize(context)
                context_lower_tokens = [w.lower() for w in context_tokens]
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)

                context_pos_set = set(context_tags)
                context_ner_set = set(context_ents)
                counter_ = Counter(context_lower_tokens)
                tf_total = len(context_lower_tokens)
                context_tf = [float(counter_[w]) / float(tf_total) for w in context_lower_tokens]
                for pos in context_pos_set :
                    pos_counter[pos] += 1
                for ner in context_ner_set :
                    ner_counter[ner] += 1
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques = pre_proc(ques)
                    ques_tokens, ques_tags, ques_ents, ques_lemmas = word_tokenize(ques)
                    ques_lower_tokens = [w.lower() for w in ques_tokens]
                    ques_chars = [list(token) for token in ques_tokens]
                    ques_lemma = {lemma if lemma != '-PRON-' else lower for lemma, lower in zip(ques_lemmas, ques_lower_tokens)}
                    ques_tf = [float(counter_[w]) / float(tf_total) for w in ques_lower_tokens]

                    ques_tokens_set = set(ques_tokens)
                    ques_lower_tokens_set = set(ques_lower_tokens)
                    match_origin = [w in ques_tokens_set for w in context_tokens]
                    match_lower = [w in ques_lower_tokens_set for w in context_lower_tokens]
                    match_lemma = [(c_lemma if c_lemma != '-PRON-' else c_lower) in ques_lemma for (c_lemma, c_lower) in zip(context_lemmas, context_lower_tokens)]

                    context_tokens_set = set(context_tokens)
                    context_lower_tokens_set = set(context_lower_tokens)
                    context_lemma = {lemma if lemma != '-PRON-' else lower for lemma, lower in zip(context_lemmas, context_lower_tokens)}
                    ques_match_origin = [w in context_tokens_set for w in ques_tokens]
                    ques_match_lower = [w in context_lower_tokens_set for w in ques_lower_tokens]
                    ques_match_lemma = [(q_lemma if q_lemma != '-PRON-' else q_lower) in context_lemma for (q_lemma, q_lower) in zip(ques_lemmas, ques_lower_tokens)]

                    ques_pos_set = set(ques_tags)
                    ques_ner_set = set(ques_ents)
                    for pos in ques_pos_set:
                        pos_counter[pos] += 1
                    for ner in ques_ner_set:
                        ner_counter[ner] += 1

                    for token in ques_tokens:
                        word_counter[token] += 1
                        ques_word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    y1sp, y2sp = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = pre_proc(answer["text"])

                        answer_start = answer['answer_start'] + len('NoAnswer ')
                        answer_end = answer_start + len(answer_text)

                        left_context = raw_context[:answer_start]
                        left_context = pre_proc(left_context)

                        mid_context = raw_context[answer_start:answer_end]

                        answer_start = len(left_context)
                        answer_end = answer_start + len(mid_context)

                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)

                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                        y1sp.append(y1)
                        y2sp.append(y2)

                    if not qa["answers"]:
                        y1s.append(0)
                        y2s.append(0)
                    if 'plausible_answers' in qa:
                        for answer in qa["plausible_answers"]:
                            answer_text = pre_proc(answer["text"])

                            answer_start = answer['answer_start'] + len('NoAnswer ')
                            answer_end = answer_start + len(answer_text)

                            left_context = raw_context[:answer_start]
                            left_context = pre_proc(left_context)

                            mid_context = raw_context[answer_start:answer_end]

                            answer_start = len(left_context)
                            answer_end = answer_start + len(mid_context)

                            answer_texts.append(answer_text)
                            answer_span = []
                            for idx, span in enumerate(spans):
                                if not (answer_end <= span[0] or answer_start >= span[1]):
                                    answer_span.append(idx)

                            y1, y2 = answer_span[0], answer_span[-1]
                            y1sp.append(y1)
                            y2sp.append(y2)
                    is_impossible = qa["is_impossible"]

                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "match_origin" : match_origin, "match_lower" : match_lower, "match_lemma" : match_lemma, "context_pos" : context_tags, "context_ner" : context_ents, "context_tf" : context_tf, "ques_tf": ques_tf,
                               "ques_tokens": ques_tokens, "ques_pos" : ques_tags, "ques_ner" : ques_ents,
                               "ques_match_origin" : ques_match_origin, "ques_match_lower" : ques_match_lower, "ques_match_lemma" : ques_match_lemma,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "y1sp": y1sp, "y2sp": y2sp, "id": total, "is_impossible": is_impossible}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        random.shuffle(examples)
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None, token2idx_dict=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                # word = "".join(array[0:-vec_size])
                word = array[0]
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.01) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    NA = "NoAnswer"
    token2idx_dict = {token: idx for idx, token in enumerate(
        embedding_dict.keys(), 3)} if token2idx_dict is None else token2idx_dict
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    token2idx_dict[NA] = 2
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    embedding_dict[NA] = [np.random.normal(scale=0.01) for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def make_dict(counter) :
    NULL = "--NULL--"
    OOV = "--OOV--"
    NA = "NoAnswer"
    index = 3
    token2idx_dict = {}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    token2idx_dict[NA] = 2
    for text in counter.keys() :
        if text not in token2idx_dict :
            token2idx_dict[text] = index
            index += 1

    return token2idx_dict


def build_features(examples, data_type, out_file, word2idx_dict, char2idx_dict, pos2idx_dict, ner2idx_dict, drop_file=None, is_test=False):

    para_limit = 450 if is_test else 450
    ques_limit = 50 if is_test else 50
    char_limit = 16

    def filter_func(example, is_test=False):
        # return len(example["context_tokens"]) > para_limit or len(example["ques_tokens"]) > ques_limit
        if not example['y2s'] or not example['y2sp']:
            return 0
        return example['y2s'][-1] >= para_limit or example['y2sp'][-1] >= para_limit

    print("Processing {} examples...".format(data_type))
    # writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0

    context_ids = []
    context_tokens = []
    context_match_origins = []
    context_match_lowers = []
    context_match_lemmas = []
    context_tfs = []
    context_char_ids = []
    context_pos_ids = []
    context_ner_ids = []
    ques_ids = []
    ques_tokens = []
    ques_match_origins = []
    ques_match_lowers = []
    ques_match_lemmas = []
    ques_char_ids = []
    ques_pos_ids = []
    ques_ner_ids = []
    ques_tfs = []
    y1 = []
    y2 = []
    y1p = []
    y2p = []
    id = []
    has_ans = []
    drop_id = []
    for example in tqdm(examples):
        total_ += 1

        if filter_func(example, is_test):
            drop_id.append(example['id'])
            continue

        len_q = len(example['ques_tokens'])
        len_q = min(len_q, ques_limit)
        pad_l = ques_limit - len_q

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        match_origin = np.zeros([para_limit], dtype=np.int32)
        match_lower = np.zeros([para_limit], dtype=np.int32)
        match_lemma = np.zeros([para_limit], dtype=np.int32)
        context_tf = np.zeros([para_limit], dtype = np.float32)
        context_pos_idxs = np.zeros([para_limit], dtype=np.int32)
        context_ner_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_match_origin = np.zeros([ques_limit], dtype=np.int32)
        ques_match_lower = np.zeros([ques_limit], dtype=np.int32)
        ques_match_lemma = np.zeros([ques_limit], dtype=np.int32)
        ques_pos_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_ner_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        ques_tf = np.zeros([ques_limit], dtype=np.int32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_pos(pos) :
            if pos in pos2idx_dict :
                return pos2idx_dict[pos]
            return 1

        def _get_ner(ner) :
            if ner in ner2idx_dict :
                return ner2idx_dict[ner]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"][:para_limit]):
            context_idxs[i] = _get_word(token)

        context_tokens.append(example["context_tokens"][:para_limit])

        for i, match in enumerate(example["match_origin"][:para_limit]) :
            match_origin[i] = 1 if match == True else 0
        for i, match in enumerate(example["match_lower"][:para_limit]) :
            match_lower[i] = 1 if match == True else 0
        for i, match in enumerate(example["match_lemma"][:para_limit]) :
            match_lemma[i] = 1 if match == True else 0

        for i, tf in enumerate(example['context_tf'][:para_limit]) :
            context_tf[i] = tf

        for i, pos in enumerate(example['context_pos'][:para_limit]) :
            context_pos_idxs[i] = _get_pos(pos)
        for i, ner in enumerate(example['context_ner'][:para_limit]) :
            context_ner_idxs[i] = _get_ner(ner)

        for j, token in enumerate(example["ques_tokens"][:ques_limit]):
            i = j + pad_l
            ques_idxs[i] = _get_word(token)

        for j, match in enumerate(example["ques_match_origin"][:ques_limit]) :
            i = j + pad_l
            ques_match_origin[i] = 1 if match == True else 0
        for j, match in enumerate(example["ques_match_lower"][:ques_limit]) :
            i = j + pad_l
            ques_match_lower[i] = 1 if match == True else 0
        for j, match in enumerate(example["ques_match_lemma"][:ques_limit]) :
            i = j + pad_l
            ques_match_lemma[i] = 1 if match == True else 0
        for j, tf in enumerate(example['ques_tf'][:ques_limit]) :
            i = j + pad_l
            ques_tf[i] = tf

        ques_token = example['ques_tokens'][:ques_limit]
        ques_tokens.append(['']*(ques_limit - len(ques_token)) + ques_token)

        for j, pos in enumerate(example['ques_pos'][:ques_limit]) :
            i = j + pad_l
            ques_pos_idxs[i] = _get_pos(pos)
        for j, ner in enumerate(example['ques_ner'][:ques_limit]) :
            i = j + pad_l
            ques_ner_idxs[i] = _get_ner(ner)

        for i, token in enumerate(example["context_chars"][:para_limit]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for j, token in enumerate(example["ques_chars"][:ques_limit]):
            i = j + pad_l
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = 0, 0
        if not example['is_impossible']:
            start, end = example["y1s"][-1], example["y2s"][-1]
            has_ans.append(1)
        else:
            has_ans.append(0)
        if not is_test:
            startp, endp = example['y1sp'][-1], example['y2sp'][-1]
        else:
            startp, endp = 0, 0

        context_ids.append(context_idxs.tolist())
        context_match_origins.append(match_origin.tolist())
        context_match_lowers.append(match_lower.tolist())
        context_match_lemmas.append(match_lemma.tolist())
        context_tfs.append(context_tf.tolist())
        context_pos_ids.append(context_pos_idxs.tolist())
        context_ner_ids.append(context_ner_idxs.tolist())
        context_char_ids.append(context_char_idxs.tolist())
        ques_ids.append(ques_idxs.tolist())
        ques_match_origins.append(ques_match_origin.tolist())
        ques_match_lowers.append(ques_match_lower.tolist())
        ques_match_lemmas.append(ques_match_lemma.tolist())
        ques_pos_ids.append(ques_pos_idxs.tolist())
        ques_ner_ids.append(ques_ner_idxs.tolist())
        ques_char_ids.append(ques_char_idxs.tolist())
        ques_tfs.append(ques_tf.tolist())
        y1.append(start)
        y2.append(end)
        y1p.append(startp)
        y2p.append(endp)
        id.append(example['id'])

    print("Build {} / {} instances of features in total".format(total, total_))

    data = {
        "context_ids" : context_ids,
        "context_tokens" : context_tokens,
        "context_match_origin" : context_match_origins,
        "context_match_lower" : context_match_lowers,
        "context_match_lemma" : context_match_lemmas,
        "context_tf" : context_tfs,
        "context_char_ids" : context_char_ids,
        "context_pos_ids" : context_pos_ids,
        "context_ner_ids" : context_ner_ids,
        "ques_ids" : ques_ids,
        "ques_match_origin": ques_match_origins,
        "ques_match_lower": ques_match_lowers,
        "ques_match_lemma": ques_match_lemmas,
        "ques_tokens" : ques_tokens,
        "ques_char_ids" : ques_char_ids,
        "ques_pos_ids" : ques_pos_ids,
        "ques_ner_ids" : ques_ner_ids,
        "ques_tf": ques_tfs,
        "y1" : y1,
        "y2" : y2,
        "y1p" : y1p,
        "y2p" : y2p,
        "id" : id,
        "total" : total,
        "has_ans": has_ans
    }
    drop_ids = {'drop_ids': drop_id}
    if drop_file:
        with open(drop_file, 'w', encoding='utf-8') as f:
            json.dump(drop_ids, f)

    with open(out_file, 'w') as f :
        json.dump(data, f)


if __name__ == '__main__' :

    save_dir = 'SQuAD/'
    word_emb_file = 'glove/glove.840B.300d.txt'

    if not os.path.isdir(save_dir) :
        os.mkdir(save_dir)

    ques_word_counter = Counter()
    word_counter, char_counter = Counter(), Counter()
    pos_counter, ner_counter = Counter(), Counter()

    train_examples, train_eval = process_file(save_dir + 'train-v2.0.json', "train", word_counter, char_counter, pos_counter, ner_counter, ques_word_counter)
    dev_examples, dev_eval = process_file(save_dir + 'dev-v2.0.json', "dev", word_counter, char_counter, pos_counter, ner_counter, ques_word_counter)

    pos2id = make_dict(pos_counter)
    ner2id = make_dict(ner_counter)

    glove_word_size = int(2.2e6)
    glove_dim = 100
    word2id = None
    word_emb, word2id = get_embedding(word_counter, "word", emb_file = word_emb_file,
                                      size = glove_word_size, vec_size = glove_dim,
                                      token2idx_dict=word2id)

    char_emb_dim = 50
    char_size = 94
    char2id = None
    char_emb, char2id = get_embedding(char_counter, "char", size = char_size, vec_size = char_emb_dim,
                                      token2idx_dict=char2id)

    build_features(train_examples, "train",
                   save_dir + 'train.json', word2id, char2id, pos2id, ner2id)
    build_features(dev_examples, "dev",
                   save_dir + 'dev.json', word2id, char2id, pos2id, ner2id, save_dir + 'drop.json', is_test=True)

    with open(save_dir + 'ques_word_counter.pkl', 'wb') as f :
        pkl.dump(ques_word_counter.most_common(), f)

    tune_idx = []
    count = 0
    for i, (word, _) in enumerate(ques_word_counter.most_common()):
        if word in word2id:
            tune_idx.append(word2id[word])
            count += 1
        if count == 1000:
            break

    with open(save_dir + 'tune_word_idx.pkl', 'wb') as f:
        pkl.dump(tune_idx, f)

    with open(save_dir + 'train_eval.json', 'w', encoding='utf-8') as f :
        json.dump(train_eval, f)
    with open(save_dir + 'dev_eval.json', 'w', encoding='utf-8') as f :
        json.dump(dev_eval, f)

    with open(save_dir + 'word_emb.json', 'w', encoding='utf-8') as f :
        json.dump(word_emb, f)

    with open(save_dir + 'word2id.json', 'w', encoding='utf-8') as f :
        json.dump(word2id, f)

    with open(save_dir + 'char2id.json', 'w', encoding='utf-8') as f :
        json.dump(char2id, f)

    with open(save_dir + 'pos2id.json', 'w') as f :
        json.dump(pos2id, f)

    with open(save_dir + 'ner2id.json', 'w') as f :
        json.dump(ner2id, f)
