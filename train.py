#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File              : train.py
# Author            : Sun Fu <cstsunfu@gmail.com>
# Date              : 23.06.2018
# Last Modified Date: 07.11.2018
# Last Modified By  : Sun Fu <cstsunfu@gmail.com>
# coding: utf-8
import argparse
import os
import torch
import pickle as pkl
import sys
import ujson as json
from model import UNet
from utils.dataset import load_data, get_batches

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default='./SQuAD/')
parser.add_argument('--model_dir', default='train_model/utf-8_new_cf',
                    help = 'path to store saved models.')
parser.add_argument('--seed', default=1023)
parser.add_argument('--use_cuda', default=True,
                    help = 'whether to use GPU acceleration.')

### parameters ###
parser.add_argument('--epochs', type = int, default=30)
parser.add_argument('--use_cf', type = bool, default = True)
parser.add_argument('--use_pd', type = bool, default = True)

parser.add_argument('--check_answer', type = bool, default=False)
parser.add_argument('--eval', type = bool, default=False)
parser.add_argument('--load_model', type = bool, default = False)
parser.add_argument('--batch_size', type = int, default=32)
parser.add_argument('--grad_clipping', type = float, default = 10)
parser.add_argument('--lrate', type = float, default=0.002)
parser.add_argument('--dropout', type = float, default=0.3)
parser.add_argument('--bound', type = float, default=0.6)
parser.add_argument('--use_char', type = bool, default=False)
parser.add_argument('--multi_point', type = bool, default=True)
parser.add_argument('--use_elmo', type = bool, default=True)
parser.add_argument('--fix_embeddings', type = bool, default=False)
parser.add_argument('--char_dim', type = int, default=50)
parser.add_argument('--pos_dim', type = int, default=12)
parser.add_argument('--ner_dim', type = int, default=8)
parser.add_argument('--evaluate', type = bool, default=False)
parser.add_argument('--char_hidden_size', type = int, default=50)
parser.add_argument('--hidden_size', type = int, default=125)
parser.add_argument('--attention_size', type = int, default=250)
parser.add_argument('--decay_period', type = int, default=10)
parser.add_argument('--decay', type = int, default=0.5)

args = parser.parse_args()
torch.manual_seed(args.seed)


def train():
    if not os.path.exists('train_model/'):
        os.makedirs('train_model/')
    if not os.path.exists('result/'):
        os.makedirs('result/')

    train_data, dev_data, word2id, id2word, char2id, opts = load_data(vars(args))
    model = UNet(opts)

    if args.use_cuda :
        model = model.cuda()

    dev_batches = get_batches(dev_data, args.batch_size, evaluation=True)

    if args.eval :
        print('load model...')
        model.load_state_dict(torch.load(args.model_dir))
        model.eval()
        model.Evaluate(dev_batches, args.data_path + 'dev_eval.json', answer_file = 'result/' + args.model_dir.split('/')[-1] + '.answers', drop_file=args.data_path + 'drop.json', dev=args.data_path + 'dev-v2.0.json')
        exit()

    if args.load_model:
        print('load model...')
        model.load_state_dict(torch.load(args.model_dir))
        model.eval()
        _, F1 = model.Evaluate(dev_batches, args.data_path + 'dev_eval.json', answer_file='result/' + args.model_dir.split('/')[-1] + '.answers', drop_file=args.data_path + 'drop.json', dev=args.data_path + 'dev-v2.0.json')
        best_score = F1
        with open(args.model_dir + '_f1_scores.pkl', 'rb') as f:
            f1_scores = pkl.load(f)
        with open(args.model_dir + '_em_scores.pkl', 'rb') as f:
            em_scores = pkl.load(f)
    else :
        best_score = 0.0
        f1_scores = []
        em_scores = []

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adamax(parameters, lr = args.lrate)

    lrate = args.lrate

    for epoch in range(1, args.epochs + 1) :
        train_batches = get_batches(train_data, args.batch_size)
        dev_batches = get_batches(dev_data, args.batch_size, evaluation=True)
        total_size = len(train_data) // args.batch_size

        model.train()
        for i, train_batch in enumerate(train_batches):
            loss = model(train_batch)
            model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, opts['grad_clipping'])
            optimizer.step()
            model.reset_parameters()

            if i % 100 == 0:
                print('Epoch = %d, step = %d / %d, loss = %.5f, lrate = %.5f best_score = %.3f' % (epoch, i, total_size, model.train_loss.value, lrate, best_score))
                sys.stdout.flush()

        model.eval()
        exact_match_score, F1 = model.Evaluate(dev_batches, args.data_path + 'dev_eval.json', answer_file = 'result/' + args.model_dir.split('/')[-1] + '.answers', drop_file=args.data_path + 'drop.json', dev=args.data_path + 'dev-v2.0.json')
        f1_scores.append(F1)
        em_scores.append(exact_match_score)
        with open(args.model_dir + '_f1_scores.pkl', 'wb') as f:
            pkl.dump(f1_scores, f)
        with open(args.model_dir + '_em_scores.pkl', 'wb') as f:
            pkl.dump(em_scores, f)

        if best_score < F1:
            best_score = F1
            print('saving %s ...' % args.model_dir)
            torch.save(model.state_dict(), args.model_dir)
        if epoch > 0 and epoch % args.decay_period == 0:
            lrate *= args.decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lrate


if __name__ == '__main__':
    train()
