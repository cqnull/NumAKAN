# -*- coding: utf-8 -*-
"""
Created on 2023/4/10

@author: QC
"""

import argparse
import copy
import logging
import math
import os
import shutil
import sys
import random
from time import strftime, localtime
import time
from tqdm import tqdm

from sklearn import metrics
import numpy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from transformers import BertModel, BertTokenizer
from torch.optim import AdamW
from torchinfo import summary
from thop import profile, clever_format

from models import NATTSYN
from utils import FinSADataset
from utils.vocab_utils import VocabHelp

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
torch.autograd.set_detect_anomaly(True)


class Instructor:
    def __init__(self, opt):
        self.opt = opt
        vocab_prefix = opt.dataset_file['dataset'][:-5]
        dep_vocab = VocabHelp.load_vocab(vocab_prefix + '_vocab_dep.vocab')
        opt.dep_vocab_size = len(dep_vocab)

        tokenizer = BertTokenizer.from_pretrained(opt.pretrained_bert_name)
        bert = BertModel.from_pretrained(opt.pretrained_bert_name, return_dict=False)
        self.model = opt.model_class(bert, opt).to(opt.device)

        tokenizer.add_tokens(['<target>', '<symbol>'], special_tokens=True)

        # if 'semeval' in opt.dataset:
        #     self.trainset = FinSADataset(opt.dataset_file['train'], opt, tokenizer, dep_vocab)
        #     self.testset = FinSADataset(opt.dataset_file['test'], opt, tokenizer, dep_vocab)
        # else:
        #     self.dataset = FinSADataset(opt.dataset_file['dataset'], opt, tokenizer, dep_vocab)
        #     testset_len = int(len(self.dataset) * opt.testset_ratio)
        #     self.trainset, self.testset = random_split(self.dataset, (len(self.dataset) - testset_len, testset_len))
        self.trainset = FinSADataset(opt.dataset_file['train'], opt, tokenizer, dep_vocab)
        self.testset = FinSADataset(opt.dataset_file['test'], opt, tokenizer, dep_vocab)

        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset) - valset_len, valset_len))
        else:
            self.valset = self.testset

        # if opt.device.type == 'cuda':
        #     logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info(
            '> n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader, test_data_loader):
        best_val_metrics = 0
        best_val_epoch = 0
        global_step = 0
        path = None

        early_stopping_rounds = 0
        total_train_time = 0
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()  # 记录开始
        torch.cuda.reset_peak_memory_stats()
        for i_epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(i_epoch))
            loss_total, n_total = 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, batch in enumerate(tqdm(train_data_loader)):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()

                inputs = [batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                start = time.time()

                if 'syn' in self.opt.model_name:
                    outputs, lex_loss, kl_loss = self.model(inputs)
                    targets = batch['label'].to(self.opt.device)
                    loss = criterion(outputs.flatten(), targets)
                    if self.opt.lex_loss:
                        loss += self.opt.r_l * lex_loss
                    elif self.opt.kl_loss:
                        loss += kl_loss
                else:
                    outputs = self.model(inputs)
                    targets = batch['label'].to(self.opt.device).float()
                    loss = criterion(outputs.flatten(), targets)
                loss.backward()
                optimizer.step()

                end = time.time()
                total_train_time += end - start

                loss_total += loss * len(outputs)
                n_total += len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_loss = loss_total / n_total
                    logger.info('loss(mse): {:.4f}'.format(train_loss))

                peak_memory = torch.cuda.max_memory_allocated()
                print(f"训练峰值显存: {peak_memory / 1024 ** 3:.2f} GB")
            eval_metrics_cos, eval_metrics_mse, eval_metrics_r2 = self._evaluate(val_data_loader)
            logger.info(
                'test_metrics > cosine: {:.4f}, mse: {:.4f}, r2: {:.4f}'.format(eval_metrics_cos, eval_metrics_mse,
                                                                                eval_metrics_r2))

            if not os.path.exists('./state_dict'):
                os.mkdir('./state_dict')
            best_model_update_flag = False
            if i_epoch == 0:
                best_val_metrics = eval_metrics_mse
                best_model_update_flag = True

            if eval_metrics_mse < best_val_metrics:
                best_model_update_flag = True
                early_stopping_rounds = 0
            # if val_metrics < best_val_metrics:
            #     best_model_update_flag = True
            if best_model_update_flag:
                best_val_metrics = eval_metrics_mse
                path = 'state_dict/{0}_{1}_val_metrics_{2}'.format(self.opt.model_name, self.opt.dataset,
                                                                   round(eval_metrics_mse, 4))

                self.best_model = copy.deepcopy(self.model)

                logger.info('>> saved: {}'.format(path))
            early_stopping_rounds += 1
            if early_stopping_rounds >= 4:
                break
        print(total_train_time)
        end_event.record()  # 记录结束
        torch.cuda.synchronize()  # 等待所有操作完成

        elapsed_time = start_event.elapsed_time(end_event)  # 返回毫秒
        print(f"总训练时间: {elapsed_time / 1000:.2f} 秒")

        torch.save(self.best_model.state_dict(), path)
        return path

    def _evaluate(self, data_loader, metrics_n=None):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        res_sentence, res_numflag, true_score, pred_score = [], [], [], []
        # switch model to evaluation mode
        self.model.eval()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        times = []
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for i_batch, t_batch in enumerate(data_loader):
                t_inputs = [t_batch[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_batch['label'].to(self.opt.device)

                flops, params = profile(self.model, inputs=(t_inputs,), verbose=False)

                # 格式化输出
                flops, params = clever_format([flops, params], "%.3f")
                print(f"模型 FLOPs: {flops}")
                print(f"模型参数量: {params}")
                starter.record()  # 记录开始时间
                if 'syn' in self.opt.model_name:
                    t_outputs, _, _ = self.model(t_inputs)
                else:
                    t_outputs = self.model(t_inputs)

                ender.record()  # 记录结束时间
                torch.cuda.synchronize()  # 等待GPU完成计算
                times.append(starter.elapsed_time(ender))  # 单位：ms

                res_sentence += t_batch['sentence']
                res_numflag += (torch.sum(t_batch['num_mask'], dim=1) > 0).tolist()
                true_score += t_targets.flatten().tolist()
                pred_score += t_outputs.flatten().tolist()

                if 'PhraseBank' in self.opt.dataset:
                    n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                    n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        # 计算统计量（均值±标准差）
        avg_time = sum(times) / len(times)
        std_time = torch.tensor(times).std().item()

        print(f"单样本平均推理时间: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"95%置信区间: {avg_time - 1.96 * std_time:.2f} ~ {avg_time + 1.96 * std_time:.2f} ms")

        eval_metrics_cos = metrics.pairwise.cosine_similarity(t_targets_all.unsqueeze(0).cpu(),
                                                              t_outputs_all.flatten().unsqueeze(0).cpu())[0][0]
        eval_metrics_mse = metrics.mean_squared_error(t_targets_all.cpu(), t_outputs_all.cpu())
        eval_metrics_r2 = metrics.r2_score(t_targets_all.cpu(), t_outputs_all.cpu())
        return eval_metrics_cos, eval_metrics_mse, eval_metrics_r2

    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']

        logger.info("bert learning rate on")
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.opt.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer

    def run(self):
        # Loss and Optimizer
        if 'PhraseBank' in self.opt.dataset:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()

        optimizer = self.get_bert_optimizer(self.model)

        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)
        # self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader, test_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        if 'PhraseBank' in self.opt.dataset:
            eval_metrics_cos, eval_metrics_mse, eval_metrics_r2 = self._evaluate(test_data_loader)
        else:
            eval_metrics_cos, eval_metrics_mse, eval_metrics_r2 = self._evaluate(test_data_loader)
        res_name = '{0}-{1}-{2}'.format(self.opt.model_name, self.opt.batch_size, self.opt.dataset)
        with open('./res.txt', 'a', encoding='utf8') as f:
            res = ' '.join(
                [res_name, str(round(float(eval_metrics_cos), 4)), str(round(float(eval_metrics_mse), 4)), '\n'])
            f.write(res)
        logger.info('#' * 60)
        logger.info(
            'best test metrics: cosine: {:.4f}, mse: {:.4f}, r2:{:.4f}'.format(eval_metrics_cos, eval_metrics_mse,
                                                                               eval_metrics_r2))


def main():
    # if os.path.exists('./state_dict/'):
    #     shutil.rmtree('./state_dict/')

    input_colses = {
        'nattsyn': ['input_ids', 'token_type_ids', 'attention_mask', 'src_mask', 'aspect_mask',
                    'num', 'num_pos', 'num_cat', 'num_mask',
                    'adj_matrix', 'asp_adj_matrix', 'lex_mat'],
    }

    model_classes = {
        'nattsyn': NATTSYN,
    }

    dataset_files = {
        'semeval_headline': {
            'train': './dataset/SemEval17-task5/headline_train.json',
            'test': './dataset/SemEval17-task5/headline_test.json',
            'dataset': './dataset/SemEval17-task5/headline.json'
        },
        'semeval_post': {
            'train': './dataset/SemEval17-task5/post_train.json',
            'test': './dataset/SemEval17-task5/post_test.json',
            'dataset': './dataset/SemEval17-task5/post.json'
        },
        'fiqa_headline': {
            'train': './dataset/fiqa/headline_train.json',
            'test': './dataset/fiqa/headline_test.json',
            'dataset': './dataset/fiqa/headline.json'
        },
        'fiqa_post': {
            'train': './dataset/fiqa/post_train.json',
            'test': './dataset/fiqa/post_test.json',
            'dataset': './dataset/fiqa/post.json'
        },
        'phrasebank': {
            'dataset': './dataset/FinancialPhraseBank-v1.0/phrasebank.json'
        }
    }

    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='nattsyn', type=str)
    parser.add_argument('--dataset', default='semeval_headline', type=str)
    parser.add_argument('--num_epoch', default=20, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=16, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=100, type=int)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--patience', default=20, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=1234, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0.2, type=float,
                        help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--testset_ratio', default=0.2, type=float)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help='initializer')

    # CNN
    parser.add_argument('--filter_sizes', default=[1, 2, 3], nargs='+', type=int, help='filter size')
    parser.add_argument('--num_filters', default=3, type=int, help='num_filters')
    parser.add_argument('--cnn_output_dim', default=32, type=int, help='cnn output dimension')
    # GCN
    parser.add_argument('--num_layers', default=2, type=int, help='num_layers')
    parser.add_argument('--attention_heads', default=1, type=int, help='attention_heads')
    parser.add_argument('--gcn_dropout', default=0.1, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    # lexicon
    parser.add_argument("--lex_loss", default=True, action='store_true', help="lex loss")
    parser.add_argument('--r_l', default=5, type=float)

    # kl
    parser.add_argument("--kl_loss", default=False, action='store_true', help="kl loss")
    parser.add_argument('--gamma', default=0.5, type=float)

    parser.add_argument("--num_shape", default=False, action='store_true', help="with num shape")
    parser.add_argument('--rho', default=0.2, type=float)

    # bert
    parser.add_argument('--pretrained_bert_name', default='./plm/finbert', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.1, help='BERT dropout rate.')
    parser.add_argument('--diff_lr', default=False, action='store_true')
    parser.add_argument('--bert_lr', default=5e-5, type=float)

    # The following parameters are only valid for the lcf-bert model
    opt = parser.parse_args()

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(opt.seed)

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    # opt.device='cpu'
    log_file = './log/{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%y%m%d_%H-%M-%S", localtime()))
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
