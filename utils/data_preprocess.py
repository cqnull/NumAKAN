# -*- coding: utf-8 -*-
"""
Created on 2023/4/26

@author: QC
"""
import html
import json
import re

import pandas as pd
from stanza.server import CoreNLPClient
from tqdm import tqdm

website_pat = re.compile(r'(http|ftp|https):\/\/[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?')
website_pat2 = re.compile(r'www.[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&:/~\+#]*[\w\-\@?^=%&/~\+#])?')
username_pat = re.compile(r'@[a-zA-Z0-9]*')
symbol_pat = re.compile(r'\$[A-Z]+')
space_pat = re.compile(r' {2,}')
abbr_repl_dict = {'don\'t': 'do not', 'can\'t': 'can not', 'EVERYONE\'S': 'EVERYONE is', 'I\'m': 'I am',
                  'let\'s': 'let us',
                  'Can\'t': 'Can not', 'Let\'s': 'Let us', 'you\'re': 'you are', 'I\'ll': 'I will',
                  'didn\'t': 'did not',
                  'Doesn\'t': 'Does not', 'we\'re': 'we are', 'it\'s': 'it is', 'won\'t': 'will not',
                  'That\'s': 'That is', 'It\'s': 'It is',
                  'they\'re': 'they are', 'Wouldn\'t': 'Would not', 'wouldn\'t': 'would not', 'i\'m': 'i am',
                  'Don\'t': 'Do not',
                  'Here\'s': 'Here is', 'What\'s': 'What is', 'that\'s': 'that is', 'Won\'t': 'Will not',
                  'I\'ve': 'I have',
                  'there\'s': 'there is', 'what\'s': 'what is', 'we\'ll': 'we will', 'isn\'t': 'is not',
                  'doesn\'t': 'does not',
                  'They\'re': 'They are', 'hadn\'t': 'had not', 'wasn\'t': 'was not', 'should\'ve': 'should have'}


def abbr_process(s):
    if '’' in s:
        s = s.replace('’', '\'')
    # if '&#39;' in s:
    #     s = s.replace('&#39;', '\'')
    # if '&quot;' in s:
    #     s = s.replace('&quot;', '')
    if '\'' in s:
        for k, v in abbr_repl_dict.items():
            s = s.replace(k, v)
    return s


def fin_data_preprocess():
    #################################################################################
    # semeval17-task5
    # headline
    # headline_tr_fpath = '../dataset/SemEval17-task5/Headline_Trainingdata.json'
    # headline_te_fpath = '../dataset/SemEval17-task5/Headlines_Testdata_withscores.json'
    # stocktwits_fpath = '../dataset/SemEval17-task5/Microblog_Trainingdata-stocktwits_full.json'
    # twitter_fpath = '../dataset/SemEval17-task5/Microblog_Trainingdata-twitter_full.json'
    # tweet_te_fpath = '../dataset/SemEval17-task5/Microblogs_Testdata_withscores.json'
    # headline_tr_df = pd.read_json(headline_tr_fpath).loc[:, ['title', 'company', 'sentiment']]
    # headline_tr_df.columns = ['sentence', 'target', 'sentiment_score']
    # headline_te_df = pd.read_json(headline_te_fpath).loc[:, ['title', 'company', 'sentiment score']]
    # headline_te_df.columns = ['sentence', 'target', 'sentiment_score']
    #
    # headline_fn_list = ['headline_train.seg', 'headline_test.seg']
    # for fn_name, headline_df in zip(headline_fn_list, [headline_tr_df, headline_te_df]):
    #     with open('../dataset/SemEval17-task5/{0}'.format(fn_name), 'w', encoding='utf8') as f:
    #         for df_idx, ele in headline_df.iterrows():
    #             ori_sentence = ele['sentence'].strip().replace('\n', ' ')
    #             target = ele['target']
    #             if 'PLC' in target:
    #                 target = target.replace('PLC', '').strip()
    #             sentiment_score = ele['sentiment_score']
    #             f.write(ori_sentence + '\n')
    #             f.write(target + '\n')
    #             f.write('%.3f' % sentiment_score + '\n')
    # with open('../dataset/SemEval17-task5/headline_train.seg', 'r', encoding='utf8') as f:
    #     tr = f.readlines()
    # with open('../dataset/SemEval17-task5/headline_test.seg', 'r', encoding='utf8') as f:
    #     te = f.readlines()
    # with open('../dataset/SemEval17-task5/headline.seg', 'w', encoding='utf8') as f:
    #     for line in tr:
    #         f.write(line)
    #     for line in te:
    #         f.write(line)
    #
    # # tweets
    # stocktwits_df = pd.read_json(stocktwits_fpath).loc[:, ['message', 'cashtag', 'sentiment score']]
    # stocktwits_df = stocktwits_df.loc[~stocktwits_df['message'].isnull()]
    # stocktwits_df['message'] = stocktwits_df['message'].apply(lambda x: x['body'])
    # stocktwits_df.columns = ['sentence', 'target', 'sentiment_score']
    #
    # twitter_df = pd.read_json(twitter_fpath).loc[:, ['text', 'cashtag', 'sentiment score']]
    # twitter_df.columns = ['sentence', 'target', 'sentiment_score']
    #
    # tweet_train_df = pd.concat([stocktwits_df, twitter_df])
    # tweet_test_df = pd.read_json(tweet_te_fpath).loc[:, ['text', 'cashtag', 'sentiment score']]
    # tweet_test_df.columns = ['sentence', 'target', 'sentiment_score']
    #
    # tweet_fn_list = ['tweet_train.seg', 'tweet_test.seg']
    # for fn_name, tweet_df in zip(tweet_fn_list, [tweet_train_df, tweet_test_df]):
    #     with open('../dataset/SemEval17-task5/{0}'.format(fn_name), 'w', encoding='utf8') as f:
    #         for df_idx, ele in tweet_df.iterrows():
    #             ori_sentence = ele['sentence'].strip().replace('\n', ' ')
    #             ori_sentence = html.unescape(ori_sentence)
    #             sentence = website_pat.sub('<WEBSITE>', ori_sentence)
    #             sentence = website_pat2.sub('<WEBSITE>', sentence)
    #             sentence = username_pat.sub('<USERNAME>', sentence)
    #             target = ele['target']
    #             sentiment_score = ele['sentiment_score']
    #             # 将其它symbol替换
    #             f.write(sentence + '\n')
    #             f.write(target + '\n')
    #             f.write('%.3f' % sentiment_score + '\n')
    # with open('../dataset/SemEval17-task5/tweet_train.seg', 'r', encoding='utf8') as f:
    #     tr = f.readlines()
    # with open('../dataset/SemEval17-task5/tweet_test.seg', 'r', encoding='utf8') as f:
    #     te = f.readlines()
    # with open('../dataset/SemEval17-task5/tweet.seg', 'w', encoding='utf8') as f:
    #     for line in tr:
    #         f.write(line)
    #     for line in te:
    #         f.write(line)
    #################################################################################
    # FiQA
    fiqa_headline_fpath = '../dataset/fiqa/raw/task1_headline_ABSA_test.json'
    fiqa_post_fpath = '../dataset/fiqa/raw/task1_post_ABSA_test.json'
    with open(fiqa_headline_fpath, 'r', encoding='utf8') as f:
        headline = json.load(f)
    with open('../dataset/fiqa/headline_test.seg', 'w', encoding='utf8') as f:
        for key, value in headline.items():
            ori_sentence = value['sentence'].strip().replace('\n', ' ')
            for item in value['info']:
                target = item['target']
                aspect = item['aspects'][2:-2].split('/')[1]
                sentiment_score = float(item['sentiment_score'])
                f.write(ori_sentence + '\n')
                f.write(target + '\n')
                f.write('%.3f' % sentiment_score + '\n')
                f.write(aspect + '\n')

    with open(fiqa_post_fpath, 'r', encoding='utf8') as f:
        post = json.load(f)
    with open('../dataset/fiqa/post_test.seg', 'w', encoding='utf8') as f:
        for key, value in post.items():
            ori_sentence = value['sentence'].strip().replace('\n', ' ')
            ori_sentence = html.unescape(ori_sentence)
            sentence = website_pat.sub('<WEBSITE>', ori_sentence)
            sentence = username_pat.sub('<USERNAME>', sentence)
            for item in value['info']:
                target = item['target']
                aspect = item['aspects'][2:-2].split('/')[1]
                sentiment_score = float(item['sentiment_score'])

                f.write(sentence + '\n')
                f.write(target + '\n')
                f.write('%.3f' % sentiment_score + '\n')
                f.write(aspect + '\n')
    #################################################################################
    # PhraseBank
    # phrasebank_fpath = '../dataset/FinancialPhraseBank-v1.0/Sentences_50Agree.txt'
    # pb_df = pd.read_csv(phrasebank_fpath, encoding="ISO-8859-1", names=['text', 'label'], delimiter='@')
    # polarity_dict = {'positive': 1, 'neutral': 0, 'negative': -1}
    # with open('../dataset/FinancialPhraseBank-v1.0/phrasebank.seg', 'w', encoding='utf8') as f:
    #     for df_idx, ele in pb_df.iterrows():
    #         ori_sentence = ele['text'].strip().replace('\n', ' ')
    #         sentiment_polarity = str(polarity_dict[ele['label']])
    #         f.write(ori_sentence + '\n')
    #         f.write(sentiment_polarity + '\n')


def fin_data_process():
    # 只经过preprocess和手动调整过的data
    f_list = [
        # '../dataset/SemEval17-task5/raw/headline_train.seg',
        # '../dataset/SemEval17-task5/raw/headline_test.seg',
        # '../dataset/SemEval17-task5/raw/tweet_train.seg',
        # '../dataset/SemEval17-task5/raw/tweet_test.seg',
        '../dataset/fiqa/raw/post_train.seg',
        '../dataset/fiqa/raw/headline_train.seg',
        '../dataset/fiqa/raw/post_test.seg',
        '../dataset/fiqa/raw/headline_test.seg',
    ]
    for raw_f in f_list:
        data_list = []
        no_target_count = 0
        with open(raw_f, 'r', encoding='utf8') as f:
            lns = f.readlines()
            if 'SemEval' in raw_f:
                step_num = 3
            elif 'fiqa' in raw_f:
                step_num = 4

            with CoreNLPClient(
                    annotators=['pos', 'depparse'],
                    # pretokenized=True,
                    timeout=30000,
                    memory='6G', be_quiet=True) as client:
                for i in tqdm(range(0, len(lns), step_num)):
                    sentence = lns[i].strip()
                    aspect_term = lns[i + 1].strip()
                    if ('fiqa' in raw_f) & ('post' in raw_f):
                        aspect_term = '$' + aspect_term

                    if 'post' in raw_f:
                        sentence = post_process(sentence, aspect_term)
                        aspect_term = '<target>'
                        if sentence is None:
                            no_target_count += 1
                            print('no target count:', no_target_count)
                            continue
                    # 如果target token中有其它字符，分割开
                    # for tok in sentence.split():
                    #     if aspect_term in tok:
                    #         if len(tok) != len(aspect_term):
                    #             redundeant_toks = tok.replace(aspect_term,' ').split()
                    #             if len(redundeant_toks)==1:
                    #                 if aspect_term[-1]==tok[-1]:
                    #                     redundeant_toks[0]
                    try:
                        aspect_term_char_from = sentence.index(aspect_term)
                    except:
                        print(sentence)
                    aspect_term_char_to = aspect_term_char_from + len(aspect_term)
                    label = lns[i + 2].strip()

                    ann = client.annotate(sentence)
                    token_list = []
                    pos_list = []
                    head_list = []
                    deprel_list = []
                    for ann_sentence in ann.sentence:
                        for token in ann_sentence.token:
                            assert token.value == token.word, print(token.value, token.word)
                            token_list.append(token.value)
                            pos_list.append(token.pos)
                            token_begin_char = token.beginChar
                            if token_begin_char == aspect_term_char_from:
                                aspect_term_token_from = token.tokenBeginIndex
                            token_end_char = token.endChar
                            if token_end_char == aspect_term_char_to:
                                aspect_term_token_to = token.tokenEndIndex
                        # 每一句单独处理head和deprel
                        sentence_token_len = len(ann_sentence.token)
                        sentence_head_list = [0] * sentence_token_len
                        sentence_deprel_list = [0] * sentence_token_len
                        for edge in ann_sentence.basicDependencies.edge:
                            source = edge.source  # head
                            target = edge.target
                            deprel = edge.dep
                            sentence_head_list[target - 1] = source
                            sentence_deprel_list[target - 1] = deprel
                        head_list.extend(sentence_head_list)
                        deprel_list.extend(sentence_deprel_list)

                    aspect_term_token_compound = ' '.join(token_list[aspect_term_token_from:aspect_term_token_to])
                    if '-' in aspect_term:
                        aspect_term_token_compound = aspect_term_token_compound.replace(' - ', '-')
                    if not aspect_term_token_compound == aspect_term:
                        print(sentence)
                        continue
                    aspect_term_list = token_list[aspect_term_token_from:aspect_term_token_to]
                    # assert ' '.join(token_list[aspect_term_token_from:aspect_term_token_to]) == aspect_term
                    assert len(token_list) == len(pos_list) == len(head_list) == len(deprel_list)
                    data_list.append({'token': token_list, 'pos': pos_list, 'head': head_list, 'deprel': deprel_list,
                                      'aspects': [{'term': aspect_term_list, 'from': aspect_term_token_from,
                                                   'to': aspect_term_token_to, 'score': label}]})
        joson_path = raw_f.replace('seg', 'json')
        joson_path = joson_path.replace('/raw', '')
        with open(joson_path, mode='w', encoding='utf-8') as f:
            json.dump(data_list, f, indent=2)


def post_process(sentence, aspect_term):
    stock_code_re = re.compile(r'\$[A-Za-z]+')

    token_list = sentence.split()
    sentence = ' '.join(token_list)

    code_list = [code_re.group() for code_re in stock_code_re.finditer(sentence)]
    for code in code_list:
        upper_code = str.upper(code)
        sentence = sentence.replace(code, upper_code)

    if aspect_term in sentence:
        sentence = sentence.replace(aspect_term, '<target>')
    else:
        return None
    for code in code_list:
        sentence = sentence.replace(code, '<symbol>')
    # 连续的'<symbol>'缩减到一个，去掉'<website>'和'<username>'
    token_list = sentence.split(' ')
    new_token_list = []
    for t_idx, token in enumerate(token_list):
        pri_token = '' if t_idx == 0 else token_list[t_idx - 1]
        if ('<symbol>' in pri_token) & (token == '<symbol>'):
            continue
        elif '<WEBSITE>' in token or '<USERNAME>' in token:
            continue
        else:
            new_token_list += [token]
    sentence = ' '.join(new_token_list)
    if '<target>' not in sentence:
        return None
    return sentence


if __name__ == '__main__':
    # fin_data_preprocess()
    fin_data_process()
