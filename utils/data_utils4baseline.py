# -*- coding: utf-8 -*-
"""
Created on 2023/4/28

@author: QC
"""

import os
import sys
from collections import Counter

import torch

from utils.tree import aspect_oriented_tree, reshape_dependency_tree, aspdep2mat, head2mat
from utils.vocab_utils import VocabHelp

sys.path.append(r'./utils/LAL-Parser/src_joint')
import re
import json
import pickle
import numpy as np
import string
from tqdm import tqdm
from transformers import BertTokenizer, BertTokenizerFast
from torch.utils.data import Dataset

num2id = {'temporal': 1, 'percentage': 2, 'monetary': 3, 'other': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10,
          '7': 11, '8': 12, '9': 13, '0': 14, '-': 15, '.': 16, 'p1': 17, 'p2': 18, 'p3': 19, 'p4': 20, 'p5': 21,
          'p6': 22, 'p7': 23, 'p8': 24, 'p9': 25}


def ParseData(data_path):
    with open(data_path) as infile:
        all_data = []
        data = json.load(infile)
        for d in data:
            text_list = d['token']  # list
            tok = ' '.join(text_list)  # word token
            length = len(text_list)  # real length
            asp_dict = d['aspects'][0]  # 只有一个aspect
            # if args.lower == True:
            # tok = [t.lower() for t in tok]
            asp = list(asp_dict['term'])  # aspect
            asp = ' '.join(asp)
            label = asp_dict['score']  # label
            pos = list(d['pos'])  # pos_tag
            head = list(d['head'])  # head
            deprel = list(d['deprel'])  # deprel
            # aspect position
            asp_from, asp_to = asp_dict['from'], asp_dict['to']
            post = [i - asp_dict['from'] for i in range(asp_dict['from'])] \
                   + [0 for _ in range(asp_dict['from'], asp_dict['to'])] \
                   + [i - asp_dict['to'] + 1 for i in range(asp_dict['to'], length)]
            # aspect mask
            mask = [0 for _ in range(asp_dict['from'])] \
                   + [1 for _ in range(asp_dict['from'], asp_dict['to'])] \
                   + [0 for _ in range(asp_dict['to'], length)]

            short = list(d['short']) if 'ssegcn' in data_path else None

            sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head, \
                      'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                      'asp_from': asp_from, 'asp_to': asp_to, 'text_list': text_list, 'short': short}
            all_data.append(sample)

    return all_data


class FinSADataset4baseline(Dataset):
    def __init__(self, fname, opt, tokenizer: BertTokenizerFast, dep_vocab: VocabHelp):
        all_data = []
        parse = ParseData
        for obj in tqdm(parse(fname), total=len(parse(fname)), desc="Training examples"):
            label = np.float32(obj['label'])
            text = obj['text']
            asp_term = obj['aspect']
            head = obj['head']
            deprel = obj['deprel']
            pos = obj['pos']
            term_start = obj['asp_from']
            term_end = obj['asp_to']
            text_list = obj['text_list']
            left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end:]
            # dualgcn
            if opt.model_name == 'dualgcn':
                from absa_parser import headparser
                headp, syntree = headparser.parse_heads(text)
                ori_adj = softmax(headp[0])
                ori_adj = np.delete(ori_adj, 0, axis=0)
                ori_adj = np.delete(ori_adj, 0, axis=1)
                ori_adj -= np.diag(np.diag(ori_adj))
                if not opt.direct:
                    ori_adj = ori_adj + ori_adj.T
                ori_adj = ori_adj + np.eye(ori_adj.shape[0])
                assert len(text_list) == ori_adj.shape[0] == ori_adj.shape[1], '{}-{}-{}'.format(len(text_list),
                                                                                                 text_list,
                                                                                                 ori_adj.shape)
                ori_depadj = ori_adj
            else:
                ori_depadj = head2mat(head)

            left_tokens, term_tokens, right_tokens = [], [], []
            left_tok2ori_map, term_tok2ori_map, right_tok2ori_map = [], [], []

            for ori_i, w in enumerate(left):
                for t in tokenizer.tokenize(w):
                    left_tokens.append(t)  # * ['expand', '##able', 'highly', 'like', '##ing']
                    left_tok2ori_map.append(ori_i)  # * [0, 0, 1, 2, 2]
            asp_start = len(left_tokens)
            offset = len(left)
            for ori_i, w in enumerate(term):
                for t in tokenizer.tokenize(w):
                    term_tokens.append(t)
                    # term_tok2ori_map.append(ori_i)
                    term_tok2ori_map.append(ori_i + offset)
            asp_end = asp_start + len(term_tokens)
            offset += len(term)
            for ori_i, w in enumerate(right):
                for t in tokenizer.tokenize(w):
                    right_tokens.append(t)
                    right_tok2ori_map.append(ori_i + offset)

            while len(left_tokens) + len(right_tokens) > opt.max_seq_len - 2 * len(term_tokens) - 3:
                if len(left_tokens) > len(right_tokens):
                    left_tokens.pop(0)
                    left_tok2ori_map.pop(0)
                else:
                    right_tokens.pop()
                    right_tok2ori_map.pop()

            bert_tokens = left_tokens + term_tokens + right_tokens

            # 拓展到bert的wordpiece上
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
            truncate_tok_len = len(bert_tokens)
            tok_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')
            asporiented_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')

            for i in range(truncate_tok_len):
                for j in range(truncate_tok_len):
                    tok_adj[i][j] = ori_depadj[tok2ori_map[i]][tok2ori_map[j]]

            context_asp_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(
                bert_tokens) + [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + [
                                  tokenizer.sep_token_id]
            context_asp_len = len(context_asp_ids)
            paddings = [0] * (opt.max_seq_len - context_asp_len)
            context_len = len(bert_tokens)
            context_asp_seg_ids = [0] * (1 + context_len + 1) + [1] * (len(term_tokens) + 1) + paddings
            src_mask = [0] + [1] * context_len + [0] * (opt.max_seq_len - context_len - 1)
            aspect_mask = [0] + [0] * asp_start + [1] * (asp_end - asp_start)
            aspect_mask = aspect_mask + (opt.max_seq_len - len(aspect_mask)) * [0]
            context_asp_attention_mask = [1] * context_asp_len + paddings
            context_asp_ids += paddings
            context_asp_ids = np.asarray(context_asp_ids, dtype='int64')
            context_asp_seg_ids = np.asarray(context_asp_seg_ids, dtype='int64')
            context_asp_attention_mask = np.asarray(context_asp_attention_mask, dtype='int64')
            src_mask = np.asarray(src_mask, dtype='int64')
            aspect_mask = np.asarray(aspect_mask, dtype='int64')
            # pad adj
            context_tok_adj = np.zeros(
                (opt.max_seq_len, opt.max_seq_len)).astype('float32')
            context_tok_adj[1:context_len + 1, 1:context_len + 1] = tok_adj
            context_asp_adj = np.zeros(
                (opt.max_seq_len, opt.max_seq_len)).astype('int64')  # 需要嵌入，用int
            context_asp_adj[1:context_len + 1, 1:context_len + 1] = asporiented_adj

            # lex
            text_list = [x.lower() for x in obj["text_list"]]
            lexicon_vector = [opt.lexicons.get(token, 0)
                              for token in text_list]
            assert len(text_list) == len(lexicon_vector)
            tok_adj = np.zeros(context_len, dtype='float32')
            for i in range(context_len):
                tok_adj[i] = lexicon_vector[tok2ori_map[i]]
            context_lex = np.zeros(opt.max_seq_len).astype('float32')
            pad_lex = np.zeros(context_asp_len).astype('float32')
            pad_lex[1:context_len + 1] = tok_adj
            context_lex[:context_asp_len] = pad_lex

            # adpt head (no-reshape)
            head_new = np.zeros(context_len, dtype='int64')
            for i in range(context_len):
                head_new[i] = head[tok2ori_map[i]]
            # pad head
            head_pad = np.zeros(opt.max_seq_len).astype('int64')
            pad_adj = np.zeros(context_asp_len).astype('int64')
            pad_adj[1:context_len + 1] = head_new
            head_pad[:context_asp_len] = pad_adj
            # pad adj
            ori_tag = [dep_vocab.stoi.get(t, dep_vocab.unk_index) for t in obj['deprel']]  # no-reshape
            tok_tag = np.zeros(context_len, dtype='int64')
            for i in range(context_len):
                tok_tag[i] = ori_tag[tok2ori_map[i]]
            context_asp_tag = np.zeros(opt.max_seq_len).astype('int64')
            pad_adj = np.zeros(context_asp_len).astype('int64')
            pad_adj[1:context_len + 1] = tok_tag
            context_asp_tag[:context_asp_len] = pad_adj
            # short mask for ssegcn
            if opt.model_name == 'ssegcn':
                row_short = obj['short']

                for i in range(context_len - 1):
                    if tok2ori_map[i + 1] == tok2ori_map[i]:
                        a = row_short[i]
                        row_short = np.insert(row_short, i, values=a, axis=0)
                column_short = row_short
                for j in range(context_len - 1):
                    if tok2ori_map[j + 1] == tok2ori_map[j]:
                        a = column_short[:, j]
                        column_short = np.insert(column_short, j, values=a, axis=1)
                mask_0 = [[-99999] * opt.max_seq_len for _ in range(opt.max_seq_len)]
                mask_1 = [[-99999] * opt.max_seq_len for _ in range(opt.max_seq_len)]
                mask_2 = [[-99999] * opt.max_seq_len for _ in range(opt.max_seq_len)]
                mask_3 = [[-99999] * opt.max_seq_len for _ in range(opt.max_seq_len)]
                mask_4 = [[-99999] * opt.max_seq_len for _ in range(opt.max_seq_len)]
                short_length = len(obj['short'])
                assert len(obj['short']) == len(obj['short'][0])
                for i in range(1):
                    for j in range(context_len):
                        mask_0[i][j] = 0
                        mask_1[i][j] = 0
                        mask_2[i][j] = 0
                        mask_3[i][j] = 0
                        mask_4[i][j] = 0
                for i in range(context_len):
                    for j in range(context_len):
                        mask_0[i + 1][j + 1] = 0
                        if column_short[i][j] == 1:
                            mask_1[i + 1][j + 1] = 0
                            mask_2[i + 1][j + 1] = 0
                            mask_3[i + 1][j + 1] = 0
                            mask_4[i + 1][j + 1] = 0
                        elif column_short[i][j] == 2:
                            mask_2[i + 1][j + 1] = 0
                            mask_3[i + 1][j + 1] = 0
                            mask_4[i + 1][j + 1] = 0
                        elif column_short[i][j] == 3:
                            mask_3[i + 1][j + 1] = 0
                            mask_4[i + 1][j + 1] = 0
                        elif column_short[i][j] == 4:
                            mask_4[i + 1][j + 1] = 0
                short_mask = np.asarray([mask_0, mask_1, mask_2, mask_3, mask_4], dtype='float32')
            else:
                short_mask = [1]

            data = {
                'sentence': text,
                # 'input_ids': ,
                # 'token_type_ids': 区分a b句，只有一句的话全为0
                # 'attention_mask': indicates to the model which tokens should be attended to, and which should not. )
                'input_ids': context_asp_ids,
                'token_type_ids': context_asp_seg_ids,
                'attention_mask': context_asp_attention_mask,
                'asp_start': asp_start,
                'asp_end': asp_end,
                'src_mask': src_mask,  # bert tokenizer后tokens的mask，去掉CLS、两个SEP和aspect term
                'aspect_mask': aspect_mask,
                ###################################################################################
                'adj_matrix': context_tok_adj,
                'asp_adj_matrix': context_asp_adj,
                ###################################################################################
                # mgfn
                'head': head_pad,
                'lex': context_lex,
                'ori_tag': context_asp_tag,  # no-reshape
                ###################################################################################
                # ssegcn
                'short_mask': short_mask,
                ###################################################################################
                'label': label,
                'aspect': asp_term
            }
            all_data.append(data)
        self.data = all_data
        # self.all_shape = list(set(all_shape_list))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def _load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, dat_fname):
    if os.path.exists(dat_fname):
        print('loading embedding_matrix:', dat_fname)
        embedding_matrix = pickle.load(open(dat_fname, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, 300))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = './dataset/glove.42B.300d.txt'
        word_vec = _load_word_vec(fname, word2idx=word2idx, embed_dim=300)
        print('building embedding_matrix:', dat_fname)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
    return embedding_matrix


def softmax(x):
    if len(x.shape) > 1:
        # matrix
        tmp = np.max(x, axis=1)
        x -= tmp.reshape((x.shape[0], 1))
        x = np.exp(x)
        tmp = np.sum(x, axis=1)
        x /= tmp.reshape((x.shape[0], 1))
    else:
        # vector
        tmp = np.max(x)
        x -= tmp
        x = np.exp(x)
        tmp = np.sum(x)
        x /= tmp
    return x


def build_senticNet():
    with open('./lexicon/senticnet.lexicon', 'rb') as l:
        lexicon_dict = pickle.load(l)
    return lexicon_dict
