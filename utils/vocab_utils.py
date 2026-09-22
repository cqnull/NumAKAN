# -*- coding: utf-8 -*-
"""
Created on 2023/5/6

@author: QC
"""

import json
import tqdm
import pickle
import argparse
import numpy as np
from collections import Counter


class VocabHelp(object):
    def __init__(self, counter, specials=['<pad>', '<self>', 'non-connect']):
        max_hop = 4
        self.pad_index = 0
        self.unk_index = 1
        counter = counter.copy()
        self.itos = list(specials)
        for tok in specials:
            del counter[tok]

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)  # words_and_frequencies is a tuple

        for word, freq in words_and_frequencies:
            self.itos.append(word)
        for hop in range(2, max_hop + 1):
            self.itos.append('ncon_' + str(hop))

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def __eq__(self, other):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def extend(self, v):
        words = v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1
        return self

    @staticmethod
    def load_vocab(vocab_path: str):
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare vocab for relation extraction.')
    parser.add_argument('--data_dir', help='TACRED directory.')
    parser.add_argument('--vocab_dir', help='Output vocab directory.')
    parser.add_argument('--lower', default=True, help='If specified, lowercase all words.')
    args = parser.parse_args()
    return args


def main():
    data_dir_dict = {'fiqa_headline': '../dataset/fiqa/',
                     'fiqa_post': '../dataset/fiqa/',
                     'semeval_headline': '../dataset/SemEval17-task5/',
                     'semeval_post': '../dataset/SemEval17-task5/',
                     }
    for fname in ['fiqa_headline', 'fiqa_post', 'semeval_headline', 'semeval_post']:
        data_dir = data_dir_dict[fname]
        # input files
        d_name = fname.split('_')[1]
        # if 'semeval' in fname:
        train_file = data_dir + d_name + '_train.json'
        test_file = data_dir + d_name + '_test.json'
        train_tokens, train_pos, train_dep, train_max_len = load_tokens(train_file)
        test_tokens, test_pos, test_dep, test_max_len = load_tokens(test_file)
        dep_counter = Counter(train_dep + test_dep)
        # else:
        #     file = data_dir + d_name + '.json'
        #     f_tokens, f_pos, f_dep, f_max_len = load_tokens(file)
        #     dep_counter = Counter(f_dep + f_dep)
        # output files
        # dep_rel
        vocab_dep_file = data_dir + d_name + '_vocab_dep.vocab'

        # build vocab
        print("building vocab...")
        dep_counter['ROOT'] = dep_counter.pop(0)
        dep_vocab = VocabHelp(dep_counter, specials=['<pad>', '<unk>', 'self', 'non-connect'])

        print("dumping to files...")
        dep_vocab.save_vocab(vocab_dep_file)


def load_tokens(filename):
    with open(filename) as infile:
        data = json.load(infile)
        tokens = []
        pos = []
        dep = []
        max_len = 0
        for d in data:
            tokens.extend(d['token'])
            pos.extend(d['pos'])
            dep.extend(d['deprel'])
            max_len = max(len(d['token']), max_len)
    print("{} tokens from {} examples loaded from {}.".format(len(tokens), len(data), filename))
    return tokens, pos, dep, max_len


if __name__ == '__main__':
    main()
