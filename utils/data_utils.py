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

            sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head, \
                      'deprel': deprel, 'length': length, 'label': label, 'mask': mask, \
                      'asp_from': asp_from, 'asp_to': asp_to, 'text_list': text_list}
            all_data.append(sample)

    return all_data


class FinSADataset(Dataset):
    def __init__(self, fname, opt, tokenizer: BertTokenizerFast, dep_vocab: VocabHelp):
        lexicon_manage = LexiconManage()
        all_data = []
        parse = ParseData
        num_max_len = 1
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
            # dep adjacency matrix process
            ori_depadj = head2mat(head)
            asporiented_dep_tag, dep_idx, dep_dir = reshape_dependency_tree(term_start, term_end, head, deprel,
                                                                            text_list)
            # asporiented_dep_tagid = [dep_vocab.stoi[t] for t in asporiented_dep_tag]
            asporiented_ori_depadj = aspdep2mat(asporiented_dep_tag, term_start, term_end, dep_vocab)
            # generate lexicon matrix
            lex_ori_mat = lexicon_manage.get_sentiment(text_list, pos)
            # 拓展到bert的wordpiece上
            tok2ori_map = left_tok2ori_map + term_tok2ori_map + right_tok2ori_map
            truncate_tok_len = len(bert_tokens)
            tok_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')
            asporiented_adj = np.zeros(
                (truncate_tok_len, truncate_tok_len), dtype='float32')
            lex_mat = np.zeros(
                (truncate_tok_len, lexicon_manage.lexicon_num), dtype='float32')
            for i in range(truncate_tok_len):
                for j in range(truncate_tok_len):
                    tok_adj[i][j] = ori_depadj[tok2ori_map[i]][tok2ori_map[j]]
                    asporiented_adj[i][j] = asporiented_ori_depadj[tok2ori_map[i]][tok2ori_map[j]]
                    lex_mat[i] = lex_ori_mat[tok2ori_map[i]]

            # 处理数字
            ori_num_list, num_list, num_cat_list, float_num_list, num_tok_idx_list, shape_list, shape_sentence = numeral_process(
                text_list, term_tokens)
            num_mask = np.zeros(opt.max_seq_len, dtype='int64')
            # num分割并与num_cat对齐
            if num_list:
                num_ids = []
                num_cat_ids = []
                num_pos_ids = []
                num_mask = []
                # 更新num_max_len
                num_len = sum([len(n) for n in num_list])
                num_max_len = num_len if num_len > num_max_len else num_max_len
                for num, num_cat in zip(num_list, num_cat_list):
                    num_char_list = list(num)
                    try:
                        num_ids = num_ids + [num2id[num_char] for num_char in num_char_list]
                    except:
                        print('num_ids here')
                        continue
                    pos_list = list(range(1, len(num_char_list) + 1))[::-1]
                    num_pos_ids += [num2id['p' + str(pos)] for pos in pos_list]
                    num_cat_ids += [num2id[num_cat]] * len(num_char_list)
                    num_mask += [1] * len(num_char_list)
            else:
                # embbeding mat中0全为0
                num_ids = [0]
                num_cat_ids = [0]
                num_pos_ids = [0]
                num_mask = [0]

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
            # pad adj and lex
            context_tok_adj = np.zeros(
                (opt.max_seq_len, opt.max_seq_len)).astype('float32')
            context_tok_adj[1:context_len + 1, 1:context_len + 1] = tok_adj
            context_asp_adj = np.zeros(
                (opt.max_seq_len, opt.max_seq_len)).astype('int64')  # 需要嵌入，用int
            context_asp_adj[1:context_len + 1, 1:context_len + 1] = asporiented_adj
            context_lex_mat = np.zeros(
                (opt.max_seq_len, lexicon_manage.lexicon_num)).astype('float32')
            context_lex_mat[1:context_len + 1] = lex_mat

            data = {
                'sentence': text,
                'bert_tokens': ' '.join(tokenizer.convert_ids_to_tokens(context_asp_ids[src_mask > 0])),
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
                'lex_mat': context_lex_mat,
                ###################################################################################
                'num': num_ids,
                'num_cat': num_cat_ids,
                'num_pos': num_pos_ids,
                'num_mask': num_mask,
                # 'num_float': float_num_list,
                ###################################################################################
                'label': label,
                'aspect': asp_term
            }
            all_data.append(data)
            # num padding
        for ele_idx, ele in enumerate(all_data):
            num = ele['num']
            num_mask = ele['num_mask']
            num_cat = ele['num_cat']
            num_pos = ele['num_pos']
            assert len(num) == len(num_mask) == len(num_cat) == len(num_pos)
            pad_num = num_max_len - len(num)
            all_data[ele_idx]['num'] = torch.LongTensor(num + [0] * pad_num)  # longtensor用于embedding
            all_data[ele_idx]['num_mask'] = torch.LongTensor(num_mask + [0] * pad_num)
            all_data[ele_idx]['num_cat'] = torch.LongTensor(num_cat + [0] * pad_num)
            all_data[ele_idx]['num_pos'] = torch.LongTensor(num_pos + [0] * pad_num)
        self.data = all_data
        # self.all_shape = list(set(all_shape_list))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def numeral_process(token_list: list, aspect_term):
    # monetary, temporal, percentage, other
    # TODO num_cat
    # num_match_iter = re.finditer(r"-?\d+[\.\\/-]?\d*", s)
    # num_match_iter = re.finditer(r"-?\.?\d+[\.\\/-]?\d*", s)
    num_matcher = re.compile(r"[-\.]?\d+[\.]?\d*")
    num_cat_list = []
    num_list = []
    num_tok_idx_list = []
    ori_num_list = []
    shape_list = []
    shape_sentence = []
    for tidx, token in enumerate(token_list):
        if token == aspect_term or token in aspect_term:
            shape_sentence += [token]
            continue
        if re.search('\d', token):
            re_num = num_matcher.findall(token)
            if len(re_num) == 1:
                if re_num[0].endswith('-'):
                    continue
                if '-' in re_num[0] and re_num[0].index('-') != 0:
                    continue
                num_list += re_num
                num_tok_idx_list.append(tidx)
            else:
                continue

            pre_tok = '' if tidx == 0 else token_list[tidx - 1]
            next_tok = '' if tidx == len(token_list) - 1 else token_list[tidx + 1]
            ori_num_list += [token]

            if '$' in token:
                num_cat_list.append('monetary')
            elif '\\' in token or '/' in token:
                num_cat_list.append('temporal')
            elif 'month' in token or 'year' in token:
                num_cat_list.append('temporal')
            elif 'month' in token or 'year' in next_tok or 'Year' in next_tok:
                num_cat_list.append('temporal')
            elif '%' in pre_tok or '%' in next_tok or '%' in token:
                num_cat_list.append('percentage')
            elif 'Percent' in next_tok:
                num_cat_list.append('percentage')
            elif 'percent' in next_tok:
                num_cat_list.append('percentage')
            elif 'pct' in next_tok:
                num_cat_list.append('percentage')
            elif sum([m in next_tok for m in ['bln', 'billion', 'Billion', 'Million', 'million']]) > 0:
                num_cat_list.append('monetary')
            elif 'M' in token or 'm' in token or 'EUR' in token:
                num_cat_list.append('monetary')
            elif next_tok == 'M' or next_tok == 'm':
                num_cat_list.append('monetary')
            elif (float(re_num[0]) > 2000) & (float(re_num[0]) < 2020):
                num_cat_list.append('temporal')
            elif '.' in token:
                num_cat_list.append('monetary')
            else:
                num_cat_list.append('other')
            num_shape = ''
            mark = num_cat_list[-1][0]
            for c in token:
                if c in string.digits:
                    num_shape += mark
                elif c in ',.+-':
                    num_shape += c
            num_shape = '[' + num_shape + ']'
            shape_list.append(num_shape)
            shape_sentence += [num_shape]
        else:
            shape_sentence += [token]
    # if num_list:
    #     for n, sap in zip(num_list, shape_list):
    #         s = s.replace(n, sap)
    shape_sentence = ' '.join(shape_sentence)
    float_num_list = [float(n) for n in num_list]
    return ori_num_list, num_list, num_cat_list, float_num_list, num_tok_idx_list, shape_list, shape_sentence


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


class LexiconManage:
    def __init__(self):
        lexicon_list = ['lm', 'senticnet', 'smsl']
        self.lexicon_num = len(lexicon_list)
        self.lexicon_dict = {}
        for lex in lexicon_list:
            with open('./lexicon/' + lex + '.lexicon', 'rb') as l:
                self.lexicon_dict[lex] = pickle.load(l)

    def get_sentiment(self, word_list, pos_list):
        senti_arr = np.zeros((len(word_list), len(self.lexicon_dict)))
        for lidx, lexicon_name in enumerate(self.lexicon_dict.keys()):
            lexicon = self.lexicon_dict[lexicon_name]
            for widx, (word, pos) in enumerate(zip(word_list, pos_list)):
                word = word.lower()
                pos = pos.upper()
                if lexicon_name == 'smsl':  # smsl有pos
                    senti_score_pos = lexicon.get((word, pos))
                    senti_score_wo_pos = lexicon.get((word, np.nan))
                    if senti_score_pos:
                        senti_arr[widx, lidx] = senti_score_pos
                    elif senti_score_wo_pos:
                        senti_arr[widx, lidx] = senti_score_wo_pos
                    else:
                        senti_arr[widx, lidx] = 0.0
                else:
                    senti_arr[widx, lidx] = lexicon.get(word, 0.0)
        return senti_arr


if __name__ == '__main__':
    # ParseData('../dataset/fiqa/post.json')
    lex_manag = LexiconManage()
    lex_manag.get_sentiment('@TradeIdea might buy $TSLA after hours if I can get in on a dip after earnings'.split(),
                            ['NN'] * 9)
