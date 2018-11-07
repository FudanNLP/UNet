#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : share_model.py
# Author            : Sun Fu <cstsunfu@gmail.com>
# Date              : 23.06.2018
# Last Modified Date: 07.11.2018
# Last Modified By  : Sun Fu <cstsunfu@gmail.com>
import torch
import numpy as np
import torch.nn as nn
import pickle as pkl
import torch.nn.functional as F
import ujson as json

from torch.autograd import Variable
from utils.layers import StackedBRNN, Dropout, FullAttention, WordAttention, Summ, PointerNet
from utils.dataset import get_data
from utils.eval2 import evaluate
from allennlp.modules.elmo import Elmo


class AverageMeter(object):
    """Keep exponential weighted averages."""
    def __init__(self, beta=0.99):
        self.beta = beta
        self.moment = 0
        self.value = 0
        self.t = 0

    def state_dict(self):
        return vars(self)

    def load(self, state_dict):
        for k, v in state_dict.items():
            self.__setattr__(k, v)

    def update(self, val):
        self.t += 1
        self.moment = self.beta * self.moment + (1 - self.beta) * val
        # bias correction
        self.value = self.moment / (1 - self.beta ** self.t)


class UNet(nn.Module):
    def __init__(self, opts):
        super(UNet, self).__init__()
        self.opts = opts
        self.build_model()
        self.bound = opts['bound']
        self.use_cf = opts['use_cf']
        self.use_pd = opts['use_pd']
        self.train_loss = AverageMeter()

    def build_model(self):
        opts = self.opts
        print('load embedding...')
        word_emb = np.array(get_data(opts['data_path'] + 'word_emb.json'), dtype=np.float32)
        word_size = word_emb.shape[0]
        word_dim = word_emb.shape[1]
        self.word_embeddings = nn.Embedding(word_emb.shape[0], word_dim, padding_idx=0)
        self.word_embeddings.weight.data = torch.from_numpy(word_emb)

        self.pos_embeddings = nn.Embedding(opts['pos_size'], opts['pos_dim'], padding_idx=0)

        self.ner_embeddings = nn.Embedding(opts['ner_size'], opts['ner_dim'], padding_idx=0)

        self.fix_embeddings = opts['fix_embeddings']

        if self.fix_embeddings :
            for p in self.word_embeddings.parameters() :
                p.requires_grad = False
        else :
            with open(opts['data_path'] + 'tune_word_idx.pkl', 'rb') as f :
                tune_idx = pkl.load(f)

            self.fixed_idx = list(set([i for i in range(word_size)]) - set(tune_idx))
            fixed_embedding = torch.from_numpy(word_emb)[self.fixed_idx]
            self.register_buffer('fixed_embedding', fixed_embedding)
            self.fixed_embedding = fixed_embedding

        pos_dim = opts['pos_dim']
        ner_dim = opts['ner_dim']
        hidden_size = opts['hidden_size']
        dropout = opts['dropout']
        attention_size = opts['attention_size']

        self.use_cuda = opts['use_cuda']
        self.use_elmo = opts['use_elmo']

        if self.use_elmo :
            elmo_dim = 1024
            options_file = "./SQuAD/elmo_options.json"
            weight_file = "./SQuAD/elmo_weights.hdf5"

            self.elmo = Elmo(options_file, weight_file, 1, dropout=0)

        feat_size = 4
        low_p_word_size = word_dim + word_dim + opts['pos_dim'] + opts['ner_dim'] + feat_size
        low_q_word_size = word_dim + opts['pos_dim'] + opts['ner_dim']

        if self.use_elmo :
            low_p_word_size += elmo_dim
            low_q_word_size += elmo_dim

        self.word_attention_layer = WordAttention(input_size = word_dim,
                                                  hidden_size = attention_size,
                                                  dropout = dropout,
                                                  use_cuda = self.use_cuda)

        self.low_cat_rnn = StackedBRNN(input_size=low_p_word_size,
                                       hidden_size=hidden_size,
                                       num_layers=1,
                                       dropout=dropout,
                                       use_cuda = self.use_cuda)
        high_p_word_size = 2 * hidden_size

        self.high_cat_rnn = StackedBRNN(input_size=high_p_word_size,
                                        hidden_size=hidden_size,
                                        num_layers=1,
                                        dropout=dropout,
                                        use_cuda = self.use_cuda)

        und_q_word_size = 2 * (2 * hidden_size)

        self.und_cat_rnn = StackedBRNN(input_size=und_q_word_size,
                                       hidden_size=hidden_size,
                                       num_layers=1,
                                       dropout=dropout,
                                       use_cuda = self.use_cuda)

        attention_inp_size = word_dim + 2 * (2 * hidden_size)

        if self.use_elmo :
            attention_inp_size += elmo_dim

        self.low_attention_layer = FullAttention(input_size = attention_inp_size,
                                                 hidden_size = attention_size,
                                                 dropout = dropout,
                                                 use_cuda = self.use_cuda)

        self.high_attention_layer = FullAttention(input_size=attention_inp_size,
                                                  hidden_size=attention_size,
                                                  dropout=dropout,
                                                  use_cuda=self.use_cuda)

        self.und_attention_layer = FullAttention(input_size=attention_inp_size,
                                                 hidden_size=attention_size,
                                                 dropout=dropout,
                                                 use_cuda=self.use_cuda)

        fuse_inp_size = 5 * (2 * hidden_size)

        self.fuse_rnn = StackedBRNN(input_size = fuse_inp_size,
                                    hidden_size = hidden_size,
                                    num_layers = 1,
                                    dropout = dropout,
                                    use_cuda=self.use_cuda)

        self_attention_inp_size = word_dim + pos_dim + ner_dim + 6 * (2 * hidden_size) + 1

        if self.use_elmo :
            self_attention_inp_size += elmo_dim

        self.self_attention_layer = FullAttention(input_size=self_attention_inp_size,
                                                  hidden_size=attention_size,
                                                  dropout=dropout,
                                                  use_cuda=self.use_cuda)

        self.self_rnn = StackedBRNN(input_size = 2 * (2 * hidden_size),
                                    hidden_size = hidden_size,
                                    num_layers = 1,
                                    dropout = dropout,
                                    use_cuda=self.use_cuda)

        if self.opts['multi_point']:
            self.self_rnn_p = StackedBRNN(input_size = 2 * (2 * hidden_size),
                                          hidden_size = hidden_size,
                                          num_layers = 1,
                                          dropout = dropout,
                                          use_cuda=self.use_cuda)

        self.summ_layer = Summ(input_size=2 * hidden_size,
                               dropout=dropout,
                               use_cuda=self.use_cuda)
        if self.opts['multi_point']:
            self.summ_layer_p = Summ(input_size=2 * hidden_size,
                                     dropout=dropout,
                                     use_cuda=self.use_cuda)
        self.summ_cf = Summ(input_size=2 * hidden_size,
                            dropout=dropout,
                            use_cuda=self.use_cuda)
        self.summ_layer2 = Summ(input_size=2 * hidden_size,
                                dropout=dropout,
                                use_cuda=self.use_cuda)
        self.summ_cf2 = Summ(input_size=2 * hidden_size,
                             dropout=dropout,
                             use_cuda=self.use_cuda)
        self.pointer_layer = PointerNet(input_size=2 * hidden_size, opt=self.opts, use_cuda=self.use_cuda)
        if self.opts['multi_point']:
            self.pointer_layer_p = PointerNet(input_size=2 * hidden_size, opt=self.opts, use_cuda=self.use_cuda)
        if self.opts['check_answer']:
            self.has_ans = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(6*hidden_size, 2))
        else:
            self.has_ans = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(6*hidden_size, 2))

    def reset_parameters(self) :
        if not self.fix_embeddings :
            self.word_embeddings.weight.data[self.fixed_idx] = self.fixed_embedding

    def compute_mask(self, x):
        mask = torch.eq(x, 0)
        if self.use_cuda:
            mask = mask.cuda()
        return mask

    def prepare_data(self, batch_data):
        """
        batch_data[0] : passage_ids,
        batch_data[1] : passage_char_ids,
        batch_data[2] : passage_pos_ids,
        batch_data[3] : passage_ner_ids,
        batch_data[4] : passage_match_origin,
        batch_data[5] : passage_match_lower,
        batch_data[6] : passage_match_lemma,
        batch_data[7] : passage_tf,
        batch_data[8] : ques_ids,
        batch_data[9] : ques_char_ids,
        batch_data[10] : ques_pos_ids,
        batch_data[11] : ques_ner_ids,
        batch_data[12] : y1,
        batch_data[13] : y2,
        batch_data[14] : id,
        batch_data[15] : passage_tokens,
        batch_data[16] : ques_tokens
        batch_data[17] : has_ans
        batch_data[18] : ques_tf
        batch_data[19] : ques_match_origin
        batch_data[20] : ques_match_lower
        batch_data[21] : ques_match_lemm
        batch_data[22] : y1p,
        batch_data[23] : y2p,
        """

        passage_ids = Variable(torch.LongTensor(batch_data[0]))
        passage_pos_ids = Variable(torch.LongTensor(batch_data[2]))
        passage_ner_ids = Variable(torch.LongTensor(batch_data[3]))
        passage_match_origin = Variable(torch.LongTensor(batch_data[4]))
        passage_match_lower = Variable(torch.LongTensor(batch_data[5]))
        passage_match_lemma = Variable(torch.LongTensor(batch_data[6]))
        passage_tf = Variable(torch.FloatTensor(batch_data[7]))
        has_ans = Variable(torch.LongTensor(batch_data[17]))

        passage_elmo_ids = batch_data[15]

        ques_ids = Variable(torch.LongTensor(batch_data[8]))
        ques_pos_ids = Variable(torch.LongTensor(batch_data[10]))
        ques_ner_ids = Variable(torch.LongTensor(batch_data[11]))
        ques_tf = Variable(torch.FloatTensor(batch_data[18]))
        ques_match_origin = Variable(torch.LongTensor(batch_data[19]))
        ques_match_lower = Variable(torch.LongTensor(batch_data[20]))
        ques_match_lemma = Variable(torch.LongTensor(batch_data[21]))

        ques_elmo_ids = batch_data[16]

        y1 = Variable(torch.LongTensor(batch_data[12]))
        y2 = Variable(torch.LongTensor(batch_data[13]))
        y1p = Variable(torch.LongTensor(batch_data[22]))
        y2p = Variable(torch.LongTensor(batch_data[23]))

        q_len = len(batch_data[21][0])
        p_lengths = passage_ids.ne(0).long().sum(1)
        q_lengths = ques_ids.ne(0).long().sum(1)

        passage_maxlen = int(torch.max(p_lengths, 0)[0])
        ques_maxlen = int(torch.max(q_lengths, 0)[0])

        passage_ids = passage_ids[:, :passage_maxlen]
        passage_pos_ids = passage_pos_ids[:, :passage_maxlen]
        passage_ner_ids = passage_ner_ids[:, :passage_maxlen]
        passage_match_origin = passage_match_origin[:, :passage_maxlen]
        passage_match_lower = passage_match_lower[:, :passage_maxlen]
        passage_match_lemma = passage_match_lemma[:, :passage_maxlen]
        passage_tf = passage_tf[:, :passage_maxlen]

        ques_elmo_ids = ques_elmo_ids[:, q_len - ques_maxlen:, :]
        ques_ids = ques_ids[:, q_len - ques_maxlen:]
        ques_pos_ids = ques_pos_ids[:, q_len - ques_maxlen:]
        ques_ner_ids = ques_ner_ids[:, q_len - ques_maxlen:]
        ques_match_origin = ques_match_origin[:, q_len - ques_maxlen:]
        ques_match_lower = ques_match_lower[:, q_len - ques_maxlen:]
        ques_match_lemma = ques_match_lemma[:, q_len - ques_maxlen:]
        ques_tf = ques_tf[:, q_len - ques_maxlen:]

        p_mask = self.compute_mask(passage_ids)
        q_mask = self.compute_mask(ques_ids)
        cat_mask = torch.cat([torch.zeros_like(q_mask).byte(), p_mask], 1)
        m = torch.zeros(q_mask.size(0)).byte().view(-1, 1).cuda()
        q_mask = torch.cat([q_mask, m], 1)

        passage_ids = torch.cat([ques_ids, passage_ids], dim=1)
        passage_pos_ids = torch.cat([ques_pos_ids, passage_pos_ids], dim=1)
        passage_ner_ids = torch.cat([ques_ner_ids, passage_ner_ids], dim=1)
        passage_match_origin = torch.cat([ques_match_origin, passage_match_origin], dim=1)
        passage_match_lower = torch.cat([ques_match_lower, passage_match_lower], dim=1)
        passage_match_lemma = torch.cat([ques_match_lemma, passage_match_lemma], dim=1)
        passage_tf = torch.cat([ques_tf, passage_tf], dim=1)

        if self.use_cuda:
            passage_ids = passage_ids.cuda()
            passage_elmo_ids = passage_elmo_ids.cuda()
            ques_elmo_ids = ques_elmo_ids.cuda()
            passage_pos_ids = passage_pos_ids.cuda()
            passage_ner_ids = passage_ner_ids.cuda()
            passage_match_origin = passage_match_origin.cuda()
            passage_match_lower = passage_match_lower.cuda()
            passage_match_lemma = passage_match_lemma.cuda()
            passage_tf = passage_tf.cuda()
            y1 = y1.cuda()
            y2 = y2.cuda()
            y1p = y1p.cuda()
            y2p = y2p.cuda()
            has_ans = has_ans.cuda()

        batch_data = {
            "passage_ids": passage_ids,
            "passage_elmo_ids" : passage_elmo_ids,
            "ques_elmo_ids" : ques_elmo_ids,
            "passage_pos_ids": passage_pos_ids,
            "passage_ner_ids": passage_ner_ids,
            "passage_match_origin": passage_match_origin.unsqueeze(2).float(),
            "passage_match_lower": passage_match_lower.unsqueeze(2).float(),
            "passage_match_lemma": passage_match_lemma.unsqueeze(2).float(),
            "passage_tf" : passage_tf.unsqueeze(2),
            "p_mask" : p_mask,
            "q_mask" : q_mask,
            "y1": y1,
            "y2": y2,
            "y1p": y1p,
            "y2p": y2p,
            "id" : batch_data[14],
            "has_ans": has_ans,
            "q_len": ques_maxlen,
            "cat_mask": cat_mask
        }

        return batch_data

    def encoding_forward(self, batch_data):

        cat_ids = batch_data['passage_ids']
        passage_elmo_ids = batch_data['passage_elmo_ids']
        ques_elmo_ids = batch_data['ques_elmo_ids']
        cat_pos_ids = batch_data['passage_pos_ids']
        cat_ner_ids = batch_data['passage_ner_ids']
        cat_match_origin = batch_data['passage_match_origin']
        cat_match_lower = batch_data['passage_match_lower']
        cat_match_lemma = batch_data['passage_match_lemma']
        cat_tf = batch_data['passage_tf']
        p_mask = batch_data['p_mask']
        q_len = batch_data['q_len']

        q_mask = batch_data['q_mask']
        cat_mask = batch_data['cat_mask']

        opts = self.opts
        dropout = opts['dropout']
        ### ElMo ###
        if self.use_elmo :
            passage_elmo = self.elmo(passage_elmo_ids)['elmo_representations'][0]
            ques_elmo = self.elmo(ques_elmo_ids)['elmo_representations'][0]
            ### embedding dropout ###
            ques_elmo = Dropout(ques_elmo, dropout, self.training, use_cuda=self.use_cuda)
            passage_elmo = Dropout(passage_elmo, dropout, self.training, use_cuda=self.use_cuda)
            cat_elmo = torch.cat([ques_elmo, passage_elmo], 1)

        ### embeddings ###
        cat_emb = self.word_embeddings(cat_ids)
        cat_pos_emb = self.pos_embeddings(cat_pos_ids)
        cat_ner_emb = self.ner_embeddings(cat_ner_ids)

        ### embedding dropout ###
        cat_emb = Dropout(cat_emb, dropout, self.training, use_cuda = self.use_cuda)

        passage_emb = cat_emb[:, q_len:]
        passage_ner_emb = cat_ner_emb[:, q_len:]
        passage_pos_emb = cat_pos_emb[:, q_len:]
        passage_match_origin = cat_match_origin[:, q_len:]
        passage_match_lemma = cat_match_lemma[:, q_len:]
        passage_match_lower = cat_match_lower[:, q_len:]
        passage_tf = cat_tf[:, q_len:]

        ques_emb = cat_emb[:, :q_len]
        ques_ner_emb = cat_ner_emb[:, :q_len]
        ques_pos_emb = cat_pos_emb[:, :q_len]
        ques_match_origin = cat_match_origin[:, :q_len]
        ques_match_lemma = cat_match_lemma[:, :q_len]
        ques_match_lower = cat_match_lower[:, :q_len]
        ques_tf = cat_tf[:, :q_len]

        ### Word Attention ###
        word_attention_outputs = self.word_attention_layer(passage_emb, p_mask, cat_emb[:, :q_len+1], q_mask, self.training)
        q_word_attention_outputs = self.word_attention_layer(cat_emb[:, :q_len+1], q_mask, passage_emb, p_mask, self.training)
        word_attention_outputs[:, 0] += q_word_attention_outputs[:, -1]
        q_word_attention_outputs = q_word_attention_outputs[:, :-1]

        p_word_inp = torch.cat([passage_emb, passage_pos_emb, passage_ner_emb, word_attention_outputs, passage_match_origin, passage_match_lower, passage_match_lemma, passage_tf], dim=2)
        q_word_inp = torch.cat([ques_emb, ques_pos_emb, ques_ner_emb, q_word_attention_outputs, ques_match_origin, ques_match_lower, ques_match_lemma, ques_tf], dim=2)

        if self.use_elmo :
            p_word_inp = torch.cat([p_word_inp, passage_elmo], dim=2)
            q_word_inp = torch.cat([q_word_inp, ques_elmo], dim=2)

        ### low, high, understanding encoding ###
        cat_word_inp = torch.cat([q_word_inp, p_word_inp], 1)
        low_cat_states = self.low_cat_rnn(cat_word_inp, cat_mask)
        # low_ques_states = self.low_ques_rnn(q_word_inp, q_mask)

        high_cat_states = self.high_cat_rnn(low_cat_states, cat_mask)
        und_cat_inp = torch.cat([low_cat_states, high_cat_states], dim=2)
        und_cat_states = self.und_cat_rnn(und_cat_inp, cat_mask)

        ### Full Attention ###

        cat_HoW = torch.cat([cat_emb, low_cat_states, high_cat_states], dim=2)

        if self.use_elmo :
            cat_HoW = torch.cat([cat_HoW, cat_elmo], dim=2)

        passage_HoW = cat_HoW[:, q_len:]
        low_passage_states = low_cat_states[:, q_len:]
        high_passage_states = high_cat_states[:, q_len:]
        und_passage_states = und_cat_states[:, q_len:]

        ques_HoW = cat_HoW[:, :q_len+1]
        low_ques_states = low_cat_states[:, :q_len+1]
        high_ques_states = high_cat_states[:, :q_len+1]
        und_ques_states = und_cat_states[:, :q_len+1]

        low_attention_outputs, low_attention_ques = self.low_attention_layer(passage_HoW, p_mask, ques_HoW, q_mask, low_ques_states, low_passage_states, self.training)
        high_attention_outputs, high_attention_ques = self.high_attention_layer(passage_HoW, p_mask, ques_HoW, q_mask, high_ques_states, high_passage_states, self.training)
        und_attention_outputs, und_attention_ques = self.und_attention_layer(passage_HoW, p_mask, ques_HoW, q_mask, und_ques_states, und_passage_states, self.training)
        low_attention_outputs[:, 0] += low_attention_ques[:, -1]
        high_attention_outputs[:, 0] += high_attention_ques[:, -1]
        und_attention_outputs[:, 0] += und_attention_ques[:, -1]

        fuse_inp = torch.cat([low_passage_states, high_passage_states, low_attention_outputs, high_attention_outputs, und_attention_outputs], dim = 2)
        fuse_ques = torch.cat([low_ques_states, high_ques_states, low_attention_ques, high_attention_ques, und_attention_ques], dim = 2)

        fuse_inp[:,0] += fuse_ques[:,-1]
        fuse_cat = torch.cat([fuse_ques[:,:-1], fuse_inp], 1)
        fused_cat_states = self.fuse_rnn(fuse_cat, cat_mask)

        ### Self Full Attention ###

        cat_HoW = torch.cat([cat_emb, cat_pos_emb, cat_ner_emb, cat_tf, fuse_cat, fused_cat_states], dim=2)

        if self.use_elmo :
            cat_HoW = torch.cat([cat_HoW, cat_elmo], dim=2)

        self_attention_outputs, _ = self.self_attention_layer(cat_HoW, cat_mask, cat_HoW, cat_mask, fused_cat_states, None, self.training)

        self_inp = torch.cat([fused_cat_states, self_attention_outputs], dim=2)

        und_cat_states = self.self_rnn(self_inp, cat_mask)
        und_passage_states_p, und_ques_states_p = None, None
        if self.opts['multi_point']:
            und_cat_states_p = self.self_rnn_p(self_inp, cat_mask)
            und_passage_states_p = und_cat_states_p[:, q_len:]
            und_ques_states_p = und_cat_states_p[:, :q_len]

        und_passage_states = und_cat_states[:, q_len:]
        und_ques_states = und_cat_states[:, :q_len]
        return und_passage_states, p_mask, und_ques_states, q_mask, und_passage_states_p, und_ques_states_p

    def decoding_forward(self, und_passage_states, p_mask, und_ques_states, q_mask, und_passage_states_p, und_ques_states_p) :

        ### ques summ vector ###

        q_summ = self.summ_layer(und_ques_states, q_mask[:, :-1], self.training)
        if self.opts['multi_point']:
            q_summ_p = self.summ_layer_p(und_ques_states_p, q_mask[:, :-1], self.training)

        ### Pointer Network ###
        logits1, logits2, _ = self.pointer_layer.forward(und_passage_states, p_mask, None, q_summ, self.training)
        logits1_p, logits2_p = None, None
        if self.opts['multi_point']:
            logits1_p, logits2_p, _ = self.pointer_layer_p.forward(und_passage_states_p, p_mask, None, q_summ_p, self.training)

        has_log = None
        if self.use_cf:
            alpha1 = F.softmax(logits1, dim=1)
            alpha2 = F.softmax(logits2, dim=1)

            p_avg1 = torch.bmm(alpha1.unsqueeze(1), und_passage_states).squeeze(1)
            p_avg2 = torch.bmm(alpha2.unsqueeze(1), und_passage_states).squeeze(1)
            p_avg = p_avg1 + p_avg2

            # alpha1_p = F.softmax(logits1_p, dim=1)
            # alpha2_p = F.softmax(logits2_p, dim=1)

            # p_avg1_p = torch.bmm(alpha1_p.unsqueeze(1), und_passage_states).squeeze(1)
            # p_avg2_p = torch.bmm(alpha2_p.unsqueeze(1), und_passage_states).squeeze(1)
            # p_avg_p = p_avg1_p + p_avg2_p

            q_summ = self.summ_layer2(und_ques_states, q_mask[:, :-1], self.training)

            first_word = und_passage_states[:, 0, :]

            # has_inp = torch.cat([p_avg, first_word, q_summ, p_avg_p], -1)
            has_inp = torch.cat([p_avg, first_word, q_summ], -1)
            has_log = self.has_ans(has_inp)
        return logits1, logits2, has_log, logits1_p, logits2_p

    def compute_loss(self, logits1, logits2, ans_log, y1, y2, has_ans, logits1_p, logits2_p, y1_p, y2_p) :

        loss1_p = loss2_p = loss1 = loss2 = loss_ans = 0

        if self.opts['multi_point']:
            loss1_p = F.cross_entropy(logits1_p, y1_p)
            loss2_p = F.cross_entropy(logits2_p, y2_p)

        if self.use_pd:
            loss1 = F.cross_entropy(logits1, y1)
            loss2 = F.cross_entropy(logits2, y2)
        if self.use_cf:
            loss_ans = F.cross_entropy(ans_log, has_ans)
        loss = loss1 + loss2 + loss_ans + loss1_p + loss2_p
        return loss

    def forward(self, batch_data):
        batch_data = self.prepare_data(batch_data)
        und_passage_states, p_mask, und_ques_states, q_mask, und_passage_states_p, und_ques_states_p = self.encoding_forward(batch_data)
        logits1, logits2, has_log, logits1_p, logits2_p = self.decoding_forward(und_passage_states, p_mask, und_ques_states, q_mask, und_passage_states_p, und_ques_states_p)

        loss = self.compute_loss(logits1, logits2, has_log, batch_data['y1'], batch_data['y2'], batch_data['has_ans'], logits1_p, logits2_p, batch_data['y1p'], batch_data['y2p'])
        self.train_loss.update(loss.data[0])
        del und_passage_states, p_mask, und_ques_states, q_mask, logits1, logits2, has_log
        return loss

    def get_predictions(self, logits1, logits2, ans_log, maxlen=15) :
        if self.use_pd:
            batch_size, P = logits1.size()
            outer = torch.matmul(F.softmax(logits1, -1).unsqueeze(2),
                                 F.softmax(logits2, -1).unsqueeze(1))

            band_mask = Variable(torch.zeros(P, P))

            if self.use_cuda :
                band_mask = band_mask.cuda()
            if self.use_cf:
                for j in range(P-1) :
                    i = j + 1
                    band_mask[i, i:max(i+maxlen, P)].data.fill_(1.0)
            else:
                for j in range(P) :
                    i = j
                    band_mask[i, i:max(i+maxlen, P)].data.fill_(1.0)

            band_mask = band_mask.unsqueeze(0).repeat(batch_size, 1, 1)
            outer = outer * band_mask

            yp1 = torch.max(torch.max(outer, 2)[0], 1)[1]
            yp2 = torch.max(torch.max(outer, 1)[0], 1)[1]
        else:
            yp1 = yp2 = torch.zeros(ans_log.size(0))

        if self.use_cf:
            sm = F.softmax(ans_log, dim=-1)
            sm[:, 0] += 0.13
            sm[:, 1] -= 0.13
            has_ans = torch.max(sm, -1)[1]
        else:
            has_ans = torch.ones(logits1.size(0))
        # if ans_log is not None:
            # has_ans = torch.zeros(ans_log.size(0)).fill_(self.bound).le(F.sigmoid(ans_log.cpu().squeeze(-1)))
        # else:
            # has_ans = torch.ones(logits1.size(0))
        return yp1, yp2, has_ans

    def convert_tokens(self, eval_file, qa_id, pp1, pp2, has_ans) :
        answer_dict = {}
        remapped_dict = {}
        for qid, p1, p2, has in zip(qa_id, pp1, pp2, has_ans) :

            p1 = int(p1)
            p2 = int(p2)
            if not int(has):
                p1 = p2 = 0
            context = eval_file[str(qid)]["context"]
            spans = eval_file[str(qid)]["spans"]
            uuid = eval_file[str(qid)]["uuid"]
            start_idx = spans[p1][0]
            end_idx = spans[p2][1]
            answer_dict[str(qid)] = context[start_idx : end_idx]
            remapped_dict[uuid] = context[start_idx : end_idx]
            if p1 == 0 and p2 == 0:
                # use_cf(not use_pd) and has_ans
                if int(has) and not self.use_pd:
                    answer_dict[str(qid)] = "HASANSWER"
                    remapped_dict[uuid] = "HASANSWER"
                else:
                    answer_dict[str(qid)] = ""
                    remapped_dict[uuid] = ""

        return answer_dict, remapped_dict

    def Evaluate(self, batches, eval_file=None, answer_file = None, drop_file=None, dev=None) :
        print('Start evaluate...')

        with open(eval_file, 'r', encoding='utf-8') as f :
            eval_file = json.load(f)
        with open(dev, 'r', encoding='utf-8') as f :
            dev = json.load(f)

        answer_dict = {}
        remapped_dict = {}

        for batch in batches :
            batch_data = self.prepare_data(batch)
            und_passage_states, p_mask, und_ques_states, q_mask, und_passage_states_p, und_ques_states_p = self.encoding_forward(batch_data)
            logits1, logits2, ans_log, logits1_p, logits2_p = self.decoding_forward(und_passage_states, p_mask, und_ques_states, q_mask, und_passage_states_p, und_ques_states_p)
            if self.opts['multi_point']:
                y1, y2, has_ans = self.get_predictions(logits1_p, logits2_p, ans_log)
            else:
                y1, y2, has_ans = self.get_predictions(logits1, logits2, ans_log)
            qa_id = batch_data['id']
            answer_dict_, remapped_dict_ = self.convert_tokens(eval_file, qa_id, y1, y2, has_ans)
            answer_dict.update(answer_dict_)
            remapped_dict.update(remapped_dict_)
            del und_passage_states, p_mask, und_ques_states, q_mask, y1, y2, answer_dict_, remapped_dict_, has_ans, ans_log, logits1, logits2

        with open(drop_file, 'r', encoding='utf-8') as f :
            drop = json.load(f)
        for i in drop['drop_ids']:
            uuid = eval_file[str(i)]["uuid"]
            answer_dict[str(i)] = ''
            remapped_dict[uuid] = ''

        with open(answer_file, 'w', encoding='utf-8') as f:
            json.dump(remapped_dict, f)
        metrics = evaluate(dev, remapped_dict)
        print("Exact Match: {}, F1: {}, Has answer F1: {}, No answer F1: {}".format(
            metrics['exact'], metrics['f1'], metrics['HasAns_f1'], metrics['NoAns_f1']))

        return metrics['exact'], metrics['f1']
