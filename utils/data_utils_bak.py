# -*- coding: utf-8 -*-
"""
Created on 2023/4/10

@author: QC
"""
import html
import json
import os
import pickle
import re
import string

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast

with open('./dataset/shape_tokens.txt', 'r', encoding='utf8') as f:
    tr = f.readlines()
shape_token_list = [shp.strip() for shp in tr]

num_max_len_dict = {}

# all_num_df = pd.read_csv('./dataset/all_num.csv')
# all_num_df.num.str.len().max()
# num_max_len = 9
num2id = {'temporal': 1, 'percentage': 2, 'monetary': 3, 'other': 4, '1': 5, '2': 6, '3': 7, '4': 8, '5': 9, '6': 10,
          '7': 11, '8': 12, '9': 13, '0': 14, '-': 15, '.': 16, 'p1': 17, 'p2': 18, 'p3': 19, 'p4': 20, 'p5': 21,
          'p6': 22, 'p7': 23, 'p8': 24, 'p9': 25}

# def vocab_rebuild():
#     with open('../plm/finbert/vocab_ori.txt', 'r', encoding='utf8') as f:
#         tr = f.readlines()
#     for s_idx, s in enumerate(shape_token_list):
#         tr[s_idx + 1] = s
#     with open('../plm/finbert/vocab.txt', 'w', encoding='utf8') as f:
#         for line in tr:
#             f.write(line)


def num_cat_process(s: str, aspect_term, dataset=None):
    # monetary, temporal, percentage, other
    # TODO num_cat
    # num_match_iter = re.finditer(r"-?\d+[\.\\/-]?\d*", s)
    # num_match_iter = re.finditer(r"-?\.?\d+[\.\\/-]?\d*", s)
    num_matcher = re.compile(r"[-\.]?\d+[\.]?\d*")
    num_cat_list = []
    num_list = []
    ori_num_list = []
    shape_list = []
    shape_sentence = []
    token_list = s.split()
    for tidx, token in enumerate(token_list):
        if token == aspect_term or token in aspect_term:
            shape_sentence += [token]
            continue
        if re.search('\d', token):
            re_num = num_matcher.findall(token)
            if re_num:
                if len(re_num) == 1:
                    if re_num[0].endswith('-'):
                        continue
                    if '-' in re_num[0] and re_num[0].index('-') != 0:
                        continue
                    num_list += re_num
                else:
                    continue
                    num_list += [str(re_num)]
            else:
                num_list += ['']

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
    return ori_num_list, num_list, num_cat_list, shape_list, shape_sentence


def build_num_embs(num_list, num_cat_list, num_embedding_matrix):
    num_one_hot_mat = np.eye(12)  # 0~9 . -
    one_hot_token = string.digits + '.' + '-'
    one_hot_ids_dict = dict(zip(one_hot_token, range(12)))

    if num_list:
        for num, num_cat in zip(num_list, num_cat_list):
            num_cat_embs = num_embedding_matrix[num2id[num_cat]]
            for nd_idx, num_digit in enumerate(num):
                digit_pos = len(num) - nd_idx
                num_digit_embs = num_one_hot_mat[one_hot_ids_dict[num_digit]]
                num_digit_pos_embs = num_embedding_matrix[num2id[digit_pos]]


class FinSADataset(Dataset):
    def __init__(self, fname, opt, tokenizer: BertTokenizerFast):
        # dataset_num_df = pd.read_csv('./dataset/{0}_num.csv'.format(opt.dataset), dtype={'num': str})
        num_max_len = 1
        dname = opt.dataset
        if 'semeval' in dname:
            step_num = 3
        elif 'fiqa' in dname:
            step_num = 3 if 'post' in dname else 4
        elif 'PhraseBank' in dname:
            step_num = 2
        else:
            raise ValueError

        with open(fname, 'r', encoding='utf8') as f:
            lns = f.readlines()
        all_data = []
        all_shape_list = []
        no_aspect_count = 0
        for i in range(0, len(lns), step_num):
            sentence = lns[i].strip()
            aspect_term = lns[i + 1].strip()
            label = lns[i + 2].strip()
            ori_num_list, num_list, num_cat_list, shape_list, shape_sentence = num_cat_process(sentence, aspect_term,
                                                                                               opt.dataset)
            if opt.num_shape:
                sentence = shape_sentence

            # if shape_list:
            #     for shape, num in zip(shape_list, num_list):
            #         sentence = sentence.replace(num, shape)

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
            # all_shape_list += shape_list
            if aspect_term not in sentence:
                raise ValueError
            try:
                bert_indices = tokenizer(sentence, aspect_term, return_tensors="pt", max_length=opt.max_seq_len,
                                         padding='max_length',
                                         truncation=True, return_offsets_mapping=True)
            except ValueError:
                print('tokenizer here')
                continue

            aspect_term_ids = np.asarray(tokenizer(aspect_term, add_special_tokens=False)['input_ids'])
            context_ids = bert_indices.input_ids[bert_indices.attention_mask > 0].numpy().ravel()
            aspect_mask_indices = search_sequence_numpy(context_ids, aspect_term_ids)[
                                  :-len(aspect_term_ids)]  # 不要最后一段
            if len(aspect_mask_indices) == 0:
                no_aspect_count += 1
                print(sentence)
                print(i)
                continue

            # assert len(aspect_mask_indices) != 0, print('no aspect term', sentence)
            aspect_mask = np.zeros(opt.max_seq_len, dtype='int')
            aspect_mask[aspect_mask_indices] = 1

            input_ids_len = bert_indices.attention_mask.sum()
            text_pair_len = bert_indices.token_type_ids.sum()
            context_len = input_ids_len - text_pair_len - 2
            src_mask = [0] + [1] * context_len + [0] * (opt.max_seq_len - context_len - 1)
            assert len(src_mask) == opt.max_seq_len
            # shape_num mask
            shape_mask = np.zeros(opt.max_seq_len, dtype='int')

            continue_flag = False
            if shape_list:
                shape_ids = tokenizer.convert_tokens_to_ids(shape_list)
                input_ids_np = bert_indices['input_ids'][0].numpy()
                for shape_id in shape_ids:
                    try:
                        id_idx = np.argwhere(input_ids_np == shape_id)[0][0]
                        shape_mask[id_idx] = 1
                    except:
                        print('shape here')
                        # print(i)
                        pass
                        # print('shape here')
                        # continue_flag = True
            if continue_flag:
                continue

            input_ids = bert_indices['input_ids'].view(-1)
            token_type_ids = bert_indices['token_type_ids'].view(-1)
            attention_mask = bert_indices['attention_mask'].view(-1)
            aspect_mask = torch.Tensor(aspect_mask)
            src_mask = torch.Tensor(src_mask)
            shape_mask = torch.Tensor(shape_mask)

            data = {
                'sentence': sentence,
                'shape_sentence': shape_sentence,
                # 'shape_list': shape_list,
                ###################################################################################
                # 'input_ids': ,
                # 'token_type_ids': 区分a b句，只有一句的话全为0
                # 'attention_mask': indicates to the model which tokens should be attended to, and which should not. )
                'input_ids': input_ids,
                'token_type_ids': token_type_ids,
                'attention_mask': attention_mask,
                'aspect_term': aspect_term,
                'aspect_mask': aspect_mask,
                'src_mask': src_mask,  # bert tokenizer后tokens的mask，去掉CLS、两个SEP和aspect term
                'shape_mask': shape_mask,
                ###################################################################################
                'num': num_ids,
                'num_cat': num_cat_ids,
                'num_pos': num_pos_ids,
                'num_mask': num_mask,
                ###################################################################################
                'label': float(label)
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
        self.all_shape = list(set(all_shape_list))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


# def collate_fn(data):
#     aspect_term = [ele['aspect_term'] for ele in data]
#     sentence = [ele['sentence'] for ele in data]
#     input_ids = torch.stack([ele['input_ids'] for ele in data])
#     token_type_ids = torch.stack([ele['token_type_ids'] for ele in data])
#     attention_mask = torch.stack([ele['attention_mask'] for ele in data])
#     label = torch.Tensor([ele['label'] for ele in data])
#     aspect_mask = torch.Tensor([ele['aspect_mask'] for ele in data])
#     src_mask = torch.Tensor([ele['src_mask'] for ele in data])
#     shape_mask = torch.Tensor([ele['shape_mask'] for ele in data])
#
#     num = [torch.LongTensor(ele['num']) for ele in data]  # 变长
#     num = torch.nn.utils.rnn.pad_sequence(num, batch_first=True, padding_value=0)
#     num_mask = [torch.LongTensor(ele['num_mask']) for ele in data]  # 变长
#     num_mask = torch.nn.utils.rnn.pad_sequence(num_mask, batch_first=True, padding_value=0)
#     num_cat = [torch.LongTensor(ele['num_cat']) for ele in data]  # 变长
#     num_cat = torch.nn.utils.rnn.pad_sequence(num_cat, batch_first=True, padding_value=0)
#     num_pos = [torch.LongTensor(ele['num_pos']) for ele in data]
#     num_pos = torch.nn.utils.rnn.pad_sequence(num_pos, batch_first=True, padding_value=0)
#
#     data = {
#         'sentence': sentence,
#         'input_ids': input_ids,
#         'token_type_ids': token_type_ids,
#         'attention_mask': attention_mask,
#         'num': num,
#         'num_mask': num_mask,
#         'num_cat': num_cat,
#         # 'num_cat_mask': num_cat_mask,
#         'num_pos': num_pos,
#         'aspect_term': aspect_term,
#         'aspect_mask': aspect_mask,
#         'src_mask': src_mask,
#         'shape_mask': shape_mask,
#         'label': label
#     }
#     return data


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


def search_sequence_numpy(arr, seq):
    Na, Nseq = arr.size, seq.size
    r_seq = np.arange(Nseq)

    M = (arr[np.arange(Na - Nseq + 1)[:, None] + r_seq] == seq).all(1)

    if M.any() > 0:
        return np.where(np.convolve(M, np.ones((Nseq), dtype=int)) > 0)[0]
    else:
        return []  # No match found


# if __name__ == '__main__':
#     sem_post_process()
# fin_data_preprocess()
# vocab_rebuild()
