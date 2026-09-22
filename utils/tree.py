# -*- coding: utf-8 -*-
"""
Created on 2023/5/2

@author: QC
"""
import numpy as np
from copy import deepcopy


def reshape_dependency_tree(as_start, as_end, head_list, deprel_list, tokens=None, multi_hop=True,
                            add_non_connect=True,
                            max_hop=4):
    '''
    Adding multi hops
    This function is at the core of our algo, it reshape the dependency tree and center on the aspect.
    In open-sourced edition, I choose not to take energy(the soft prediction of dependency from parser)
    into consideration. For it requires tweaking allennlp's source code, and the energy is space-consuming.
    And there are no significant difference in performance between the soft and the hard(with non-connect) version.

    [deprel, source(head), target]
    "dependencies": [["nn", 2, 1], ["nsubj", 3, 2], ["root", 0, 3], ["advmod", 5, 4], ["advmod", 3, 5], ["punct", 5, 6], ["advmod", 8, 7], ["advmod", 5, 8],
                    ["prep", 8, 9], ["num", 11, 10], ["pobj", 9, 11], ["prep", 9, 12], ["num", 14, 13], ["pobj", 12, 14], ["punct", 3, 15]],
    '''
    dep_tag = []
    dep_idx = []  # dep_tag的主体在token中的index，由于搜索时不考虑方向，head或target都有可能，
    dep_dir = []
    deprel_list = np.asarray(deprel_list)
    deprel_list[deprel_list == '0'] = 'root'
    dependencies = list(zip(deprel_list, head_list, list(range(1, len(deprel_list) + 1))))
    # 1 hop
    for i in range(as_start, as_end):
        for dep in dependencies:
            # 如果asp的idx是head
            if i == dep[1] - 1:  # dep parser中，root的index为0，所以token中index从1开始，减去1与as_start对齐
                # not root, not aspect 而且不在dep_idx中，一个token不能有多种hop
                if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and dep[2] != 0 and dep[2] - 1 not in dep_idx:
                    # 不要标点
                    if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(1)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[2] - 1)
            # 如果asp的idx是target
            elif i == dep[2] - 1:
                # not root, not aspect
                if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and dep[1] != 0 and dep[1] - 1 not in dep_idx:
                    if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                        dep_tag.append(dep[0])
                        dep_dir.append(2)
                    else:
                        dep_tag.append('<pad>')
                        dep_dir.append(0)
                    dep_idx.append(dep[1] - 1)

    if multi_hop:
        current_hop = 2
        added = True
        while current_hop <= max_hop and len(dep_idx) < len(tokens) and added:
            added = False
            dep_idx_temp = deepcopy(dep_idx)
            for i in dep_idx_temp:
                for dep in dependencies:
                    if i == dep[1] - 1:
                        # not root, not aspect
                        if (dep[2] - 1 < as_start or dep[2] - 1 >= as_end) and \
                                dep[2] != 0 and dep[2] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[2] - 1] not in stopWords
                                dep_tag.append('ncon_' + str(current_hop))
                                dep_dir.append(1)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[2] - 1)
                            added = True
                    elif i == dep[2] - 1:
                        # not root, not aspect
                        if (dep[1] - 1 < as_start or dep[1] - 1 >= as_end) and \
                                dep[1] != 0 and dep[1] - 1 not in dep_idx:
                            if str(dep[0]) != 'punct':  # and tokens[dep[1] - 1] not in stopWords
                                dep_tag.append('ncon_' + str(current_hop))
                                dep_dir.append(2)
                            else:
                                dep_tag.append('<pad>')
                                dep_dir.append(0)
                            dep_idx.append(dep[1] - 1)
                            added = True
            current_hop += 1

    if add_non_connect:
        for idx, token in enumerate(tokens):
            # 没有连接到aspect且不在aspect的index中
            if idx not in dep_idx and (idx < as_start or idx >= as_end):
                dep_tag.append('non-connect')
                dep_dir.append(0)
                dep_idx.append(idx)

    # add aspect and index, to make sure length matches len(tokens)
    for idx, token in enumerate(tokens):
        # 剩下的填充为<pad>，也就是aspect
        if idx not in dep_idx:
            dep_tag.append('<pad>')
            dep_dir.append(0)
            dep_idx.append(idx)

    # 根据dep_tag的主体在token中的index（dep_idx）对dep_tag等进行重排序，排序成与token对应的顺序
    index = [i[0] for i in sorted(enumerate(dep_idx), key=lambda x: x[1])]
    dep_tag = [dep_tag[i] for i in index]
    dep_idx = [dep_idx[i] for i in index]
    dep_dir = [dep_dir[i] for i in index]

    assert len(tokens) == len(dep_idx), 'length wrong'
    return dep_tag, dep_idx, dep_dir


def aspdep2mat(dep_tag, as_start, as_end, dep_vocab):
    adjmat = [[dep_vocab.stoi['non-connect']] * len(dep_tag) for _ in range(len(dep_tag))]
    for i in range(len(dep_tag)):
        for j in range(len(dep_tag)):
            if i in range(as_start, as_end) and j in range(as_start, as_end):
                adjmat[i][j] = dep_vocab.stoi[dep_tag[i]]
            elif i in range(as_start, as_end):
                adjmat[i][j] = dep_vocab.stoi[dep_tag[j]]
            elif j in range(as_start, as_end):
                adjmat[i][j] = dep_vocab.stoi[dep_tag[i]]
            elif i == j:
                adjmat[i][j] = dep_vocab.stoi['self']
    return adjmat


def simple_watch(index_dict, stoi, default_key=None):
    if default_key is not None:
        child = index_dict[default_key]
        for key in child.keys():
            print("{'%s':%d}" % (stoi[key], child[key]), end=' ')
    else:
        for child in index_dict:
            for key in child.keys():
                print("{'%s':%d}" % (stoi[key], child[key]), end=' ')
            print('', end='   ')
    print()


def head2mat(head):
    adjmat = np.zeros((len(head), len(head)))
    for tokidx, tok_head in enumerate(head):
        if tok_head == 0:
            continue  # skip ROOT
        head_idx = tok_head - 1
        adjmat[tokidx, head_idx] = 1
        adjmat[head_idx, tokidx] = 1
    np.fill_diagonal(adjmat, 1)
    return adjmat


def aspect_oriented_tree(opt, token, head, as_start, as_end):
    '''
    generate distance based weighted matrix
    :param opt: 命令行参数
    :param token: 单词列表，即一句话被tokenize后的列表
    :param head: 依赖关系
    :param as_start: aspect起始位置
    :param as_end: aspect终止位置
    :return: 该条数据的distance based weighted matrix
    '''

    stoi = {}  # 索引对单词的词典
    for i, t in enumerate(token):
        stoi[i] = t
    # print(stoi)
    children = [{}] * len(token)  # 存储每个单词与其相连的单词的索引

    # 查找连接的单词
    for i in range(len(token)):
        for j in range(len(head)):
            # 如果第j个单词的头部是第i个单词，且第j个单词不在第i个单词所属的连接单词字典里，还要不是根节点
            if head[j] - 1 == i and j not in children[i].keys() and head[j] != 0:
                children[i][j] = 1
                children[j][i] = 1  # 在第j个单词的字典中添加与第i个单词的距离，意思是对称
        # 如果第i个单词的头部所指的单词不在字典当中，且第i个单词的头部不是根节点
        if head[i] - 1 not in children[i].keys() and head[i] != 0:
            children[i][head[i] - 1] = 1
            children[head[i] - 1][i] = 1  # 在第i个单词头部所指的单词的字典中添加与第i个单词的距离，表达对称
    # simple_watch(children, stoi)

    # 计算每个单词与aspect的距离
    children_asp_all = []
    for asp_idx in range(as_start, as_end):
        children_asp = deepcopy(children)
        head_idx = list(children_asp[asp_idx].keys())  # 与aspect相连的单词集合
        head_stack = deepcopy(head_idx)  # 栈
        # print(head_idx)
        # 所有的单词还没有遍历完，并且栈中还依然有单词的话，才会接着遍历。
        # 防止出现aspect不是与所有单词相连的，栈中已经没有单词了，但是还在循环
        while (len(head_idx) < len(token)) and (len(head_stack) > 0):
            idx_in_sent = head_stack.pop(0)  # 栈顶单词
            ids = list(children_asp[idx_in_sent].keys())  # 栈顶单词的相连单词
            for idx in ids:
                # 如果索引为idx的单词不在存储与aspect相连单词的距离的字典中，且该单词不在aspect的范围里
                # head_idx还有一个作用就是判断是否被遍历过，这里没有考虑单词是否在aspect范围中，因为这样会影响接下来的计算
                if idx not in head_idx and idx != asp_idx:
                    # 把存储aspect与各个单词距离的字典更新，添加上与索引为idx的单词的距离
                    children_asp[asp_idx][idx] = children_asp[idx_in_sent][idx] + children_asp[asp_idx][idx_in_sent]
                    head_stack = [idx] + head_stack  # 更新栈，将新的单词索引放到栈顶
                    head_idx += [idx]  # 添加遍历过的单词索引到head_idx中
        # simple_watch(children_asp, stoi, asp_idx)
        children_asp_all.append(children_asp)

    # distance based weighted matrix，分为两种模式
    '''
    第一种模式：将多个aspect一起考虑。对于某个单词，查找并存储该单词与每个aspect的距离，取最小的距离为该单词到所有aspect的距离。
              在矩阵中表示为每个aspect与该单词的距离都是相同的，且是最短距离。不同aspect之间的距离设置为1，对同一个aspect距离同样是1(self-loop)

    第二种模式：分别考虑aspect。对于某个单词，查找并存储该单词与每个aspect的距离，矩阵对应位置上存储每个aspect和该单词的距离。
              不同aspect之间的距离不做特殊要求，该是啥是啥，对同一个aspect距离是1(self-loop)
    '''
    if 'bert' in opt.model_name:
        dm = np.ones((len(token), len(token))) * (np.inf)  # 初始化distance based weighted matrix，每个元素是无穷大，方便后面比较大小
    else:
        dm = np.ones((opt.max_length, opt.max_length)) * (np.inf)
    if True:
        # 第一种模式
        aspect_indices = list(range(as_start, as_end))  # aspect的索引列表
        # 第一层循环含义是遍历除单个aspect以外单词的索引
        for word_id in range(len(token)):
            distances = [np.inf]  # 存储在该单词位置上的距离，先有一个无穷大是因为dm中初始值为无穷大
            # 遍历aspect的索引，同时还需要索引的索引，因为children_asp_all的维度是aspect的数量，跟aspect的索引不匹配
            for child_id, asp_id in enumerate(aspect_indices):
                asp_child = children_asp_all[child_id][asp_id]  # 找到该aspect下跟其他所有单词的距离词典
                # 在这里有个try的原因是担心有的aspect跟某个单词不连接但是其他aspect是连接的，如果不连接的话，还是把dm中的值(无穷大)放过去
                try:
                    distances.append(asp_child[word_id])
                except:
                    distances.append(np.inf)
            real_distance = min(distances)  # 取最小值
            for asp_id in aspect_indices:
                dm[asp_id][word_id] = real_distance  # 在对应位置上赋值
                dm[word_id][asp_id] = deepcopy(dm[asp_id][word_id])  # 对称
        for asp_id in aspect_indices:
            for asp_mutual in aspect_indices:
                dm[asp_id][asp_mutual] = 1  # 不光跟其他aspect距离是1，跟自己也是1

    else:
        # 第二种模式
        aspect_indices = list(range(as_start, as_end))
        for child_id, asp_id in enumerate(aspect_indices):
            asp_child = children_asp_all[child_id][asp_id]
            word_indices = list(asp_child.keys())
            # 在这里就不需要比较大小，直接赋值即可
            for word_id in word_indices:
                dm[asp_id][word_id] = asp_child[word_id]
                dm[word_id][asp_id] = deepcopy(dm[asp_id][word_id])
        for asp_id in aspect_indices:
            dm[asp_id][asp_id] = 1  # self-loop，自己跟自己的距离才是1

    # self-loop
    for i in range(len(dm)):
        dm[i][i] = 1

    return dm
