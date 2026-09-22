# -*- coding: utf-8 -*-
"""
Created on 2023/4/19

@author: QC
"""
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import num2id, build_embedding_matrix


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        # broadcast
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class NATTSYN(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn_model = GCNAbsaModel(bert, opt=opt)
        self.numcnn_model = NumCNNModel(opt=opt)
        cnn_output_dim = opt.num_filters * len(opt.filter_sizes)
        self.classifier = nn.Linear(opt.bert_dim * 3 + cnn_output_dim, 1)
        # self.classifier = nn.Linear(opt.bert_dim * 3, 1)
        # dep_output变换
        self.v_linear = nn.Linear(in_features=opt.bert_dim,
                                  out_features=1,
                                  bias=False)


    def forward(self, inputs):

        outputs_syn_mask, outputs_sem_mask, outputs_syn, outputs_sem, sem_adj, pooled_output, bert_output, kl_loss = self.gcn_model(
            inputs)
        num_output = self.numcnn_model(inputs)
        final_outputs = torch.cat((outputs_syn_mask, outputs_sem_mask, pooled_output, num_output), dim=-1)
        # final_outputs = torch.cat((outputs_syn_mask, outputs_sem_mask, pooled_output), dim=-1)
        logits = self.classifier(final_outputs)

        lexicon_loss = None

        return logits, lexicon_loss, kl_loss


class GCNAbsaModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.gcn = GCNBert(bert, opt, opt.num_layers)

    def forward(self, inputs):
        """
        ['input_ids', 'token_type_ids', 'attention_mask', 'src_mask', 'aspect_mask',
         'num', 'num_pos', 'num_cat', 'num_mask', 'shape_mask',
         'context_tok_adj', 'asp_adj_matrix', 'lex_mat'],
        """
        input_ids, token_type_ids, attention_mask, src_mask, aspect_mask, num, num_pos, num_cat, num_mask, \
            dep_adj, asp_adj, lex_mat = inputs
        # outputs_sem，outputs_dep：两个GCN的输出，sem_adj：多头平均后的注意力矩阵，pooled_output：bert输出，对[CLS]线性变换
        # outputs_sem, outputs_dep, sem_adj, pooled_output = self.gcn(inputs)
        h1, h2, gcn_inputs, sem_adj, pooled_output, kl_loss = self.gcn(inputs)

        # avg pooling asp feature
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim)
        bert_output = (gcn_inputs * aspect_mask).sum(dim=1) / asp_wn

        ########################################################################
        # dualgcn
        # avg pooling asp feature
        output = (h1 * aspect_mask).sum(dim=1) / asp_wn
        output_sem = (h2 * aspect_mask).sum(dim=1) / asp_wn
        ########################################################################
        # mgfn
        # mask = copy.deepcopy(src_mask)
        # mask[:, 0] = 1
        # mask = mask.unsqueeze(-1).repeat(1, 1, self.opt.bert_dim)
        # h_syn = h1 * mask
        # h_sem = h2 * mask
        #
        # beta_mat = torch.matmul(h_syn, h_sem.transpose(1, 2))  # [B,S,dim] [B,dim,S] => [B,S,S]
        # beta = beta_mat.sum(1, keepdim=True)
        # attn = F.softmax(beta, dim=2)  # [B,1,S]
        # output = torch.matmul(attn, h2).squeeze(1)  # [B,dim]
        # output_sem = (h2 * aspect_mask).sum(dim=1) / (asp_wn + 1e-9)
        ########################################################################
        # attention retrival
        # outputs_syn = h1 * aspect_mask
        # outputs_sem = h2 * aspect_mask
        # alpha_mat = torch.matmul(outputs_syn, gcn_inputs.transpose(1, 2))
        # alpha = F.softmax(alpha_mat.sum(1, keepdim=True), dim=2)
        #
        # output = torch.matmul(alpha, gcn_inputs).squeeze(1)  # gcn_inputs也就是bert的输出
        # output_sem = outputs_sem.sum(dim=1)
        ########################################################################

        return output, output_sem, h1, h2, sem_adj, pooled_output, bert_output, kl_loss


class GCNBert(nn.Module):
    def __init__(self, bert, opt, num_layers):
        super(GCNBert, self).__init__()
        self.bert = bert
        self.opt = opt
        self.layers = num_layers  # opt.num_layers GCN层数
        self.mem_dim = opt.bert_dim  # GCN维度
        self.attention_heads = opt.attention_heads  # 注意力头数量
        self.bert_dim = opt.bert_dim  # 768
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.gcn_drop = nn.Dropout(opt.gcn_dropout)
        self.layernorm = LayerNorm(opt.bert_dim)
        # asp_adj embbeding
        self.asp_adj_embedding = nn.Embedding(opt.dep_vocab_size, 300, padding_idx=0)
        self.fc1 = nn.Linear(300, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1)

        self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.bert_dim if layer == 0 else self.mem_dim
            self.W.append(nn.Linear(input_dim, self.mem_dim))

        self.attn = MultiHeadAttention(opt.attention_heads, self.bert_dim)


        # lex_mat 变换
        lexicon_size = 3
        self.attention_size = self.bert_dim
        self.gate = nn.Linear(lexicon_size, self.attention_size)
        self.sigmoid = nn.Sigmoid()
        self.v = nn.Linear(self.attention_size, 1)
        # self.attention = SelfAttention(attention_size=self.attention_size,
        #                                dropout=self.att_dropout)

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask, src_mask, aspect_mask, num, num_pos, num_cat, num_mask, \
            dep_adj, asp_adj_ori, lex_mat = inputs
        src_mask = src_mask.unsqueeze(-2)  # 有原文token的mask，在-2加一维，size:n_batch*bert_max_len(85)
        # 直接call会调用forward
        sequence_output, pooled_output = self.bert(input_ids, attention_mask=attention_mask,
                                                   token_type_ids=token_type_ids, return_dict=False)
        sequence_output = self.layernorm(sequence_output)
        gcn_inputs = self.bert_drop(sequence_output)
        # gcn_inputs = sequence_output
        pooled_output = self.pooled_drop(pooled_output)

        # asp_adj 嵌入和变换
        asp_adj = self.asp_adj_embedding(asp_adj_ori)  # ori-asp-dep-adj
        asp_adj = self.fc1(asp_adj)  # [B,S,64]
        asp_adj = self.relu(asp_adj)
        asp_adj = self.fc2(asp_adj)  # (B, S, 1)
        asp_adj = asp_adj.squeeze(3)
        # 语法邻接矩阵
        syn_adj = dep_adj * asp_adj
        # syn_adj = dep_adj

        # aspect embedding
        aspect_count = aspect_mask.sum(axis=1).unsqueeze(-1)
        aspect_count[aspect_count == 0] = -1
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, 768)
        aspect_emb = (sequence_output * aspect_mask).sum(axis=1) / aspect_count

        lex_att_score = self.v(self.sigmoid(self.gate(lex_mat)) * gcn_inputs).squeeze()  # bs * max_len * 1
        lex_att_score = lex_att_score.masked_fill(src_mask.squeeze() == 0, -1e9)
        lex_att_score = F.softmax(lex_att_score, dim=-1)
        outputs_sem = (gcn_inputs * lex_att_score.unsqueeze(-1))
        attn_tensor = self.attn(outputs_sem, outputs_sem, aspect_emb, src_mask)

        # attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        attn_adj_list = [attn_tensor[:, i] for i in range(self.attention_heads)]
        # 语义邻接矩阵
        sem_adj = None

        # * Average Multi-head Attention matrixes
        for i in range(self.attention_heads):
            if sem_adj is None:
                sem_adj = attn_adj_list[i]
            else:
                sem_adj += attn_adj_list[i]
        sem_adj /= self.attention_heads

        for j in range(sem_adj.size(0)):
            sem_adj[j] -= torch.diag(torch.diag(sem_adj[j]))
            sem_adj[j] += torch.eye(sem_adj[j].size(0)).cuda()
        sem_adj = src_mask.transpose(1, 2) * sem_adj  # 再mask一下，最终只取原来att_mat的[src_mask==1,src_mask==1]部分

        outputs_sem = torch.bmm(sem_adj, outputs_sem)

        # lex_att_score = self.v(self.sigmoid(self.gate(lex_mat)) * outputs_sem).squeeze()  # bs * max_len * 1
        # lex_att_score = lex_att_score.masked_fill(src_mask.squeeze() == 0, -1e9)
        # lex_att_score = F.softmax(lex_att_score, dim=-1)
        # outputs_sem = (outputs_sem * lex_att_score.unsqueeze(-1)).sum(1)

        kl_loss = None

        denom_syn = syn_adj.sum(2).unsqueeze(2) + 1
        outputs_syn = gcn_inputs
        # *********begin multul******
        for l in range(self.layers):
            # ************SynGCN*************
            Ax_dep = syn_adj.bmm(outputs_syn)
            AxW_dep = self.W[l](Ax_dep)
            AxW_dep = AxW_dep / denom_syn
            gAxW_dep = F.relu(AxW_dep)
            outputs_syn = self.gcn_drop(gAxW_dep) if l < self.layers - 1 else gAxW_dep
        # outputs_ag，outputs_dep：两个GCN的输出，sem_adj：多头平均后的注意力矩阵，pooled_output：bert输出，对[CLS]线性变换
        return outputs_syn, outputs_sem, gcn_inputs, sem_adj, pooled_output, kl_loss


class NumCNNModel(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.filter_sizes = opt.filter_sizes
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, opt.num_filters, (k, opt.embed_dim * 3)) for k in opt.filter_sizes])
        self._num_embedding_init()

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask, src_mask, aspect_mask, num, num_pos, num_cat, num_mask, \
            dep_adj, asp_adj, lex_mat = inputs
        cnn_out = torch.cat([self.num_embedding(num),
                             self.num_embedding(num_pos),
                             self.num_embedding(num_cat)
                             ],
                            dim=2)
        cnn_out = cnn_out.unsqueeze(1)
        cnn_out = torch.cat([self.conv_and_pool(cnn_out, conv) for conv in self.convs], 1)
        return cnn_out

    def _num_embedding_init(self):
        # num_embedding_matrix = build_embedding_matrix(
        #     word2idx=num2id,
        #     dat_fname='./num_embedding_matrix.dat')
        # self.num_embedding = nn.Embedding.from_pretrained(torch.tensor(num_embedding_matrix, dtype=torch.float),
        #                                                   freeze=False).to(self.opt.device)
        self.num_embedding = nn.Embedding(len(num2id) + 2, 300, padding_idx=0)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        """

        Args:
            h: 头数量
            d_model: self.attdim = 100
            dropout:
        """
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        # self.dropout = None
        # self.weight_m = nn.Parameter(torch.zeros(self.h, self.d_k, self.d_k))
        # self.bias_m = nn.Parameter(torch.zeros(1))
        self.weight_m = nn.Linear(self.d_k, self.d_k, bias=False)
        self.bias_m = nn.Parameter(torch.zeros(1))
        self.dense = nn.Linear(d_model, self.d_k)

    def forward(self, query, key, aspect, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        batch, aspect_dim = aspect.size()[0], aspect.size()[1]
        aspect = aspect.unsqueeze(1).expand(batch, self.h, aspect_dim)
        aspect = self.dense(aspect)
        aspect = aspect.unsqueeze(2).expand(batch, self.h, query.size()[2], self.d_k)
        attn = attention(query, key, aspect, self.weight_m, self.bias_m, mask=mask, dropout=self.dropout)
        return attn


def attention(query, key, aspect, weight_m, bias_m, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # aspect_scores = torch.tanh(torch.add(torch.matmul(aspect, weight_m(key).transpose(-2, -1)), bias_m))
    # aspect_scores = torch.tanh(
    #     torch.add(torch.matmul(torch.matmul(aspect, weight_m), key.transpose(-2, -1)), bias_m))

    # scores = torch.add(scores, aspect_scores)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1).clone()
    if dropout is not None:
        p_attn = dropout(p_attn)
    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
