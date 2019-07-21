# -*- coding: utf-8 -*-
# @Time    : 2019/7/14 23:41
# @Author  : kean
# @Email   : ?
# @File    : utils.py
# @Software: PyCharm


import os
import re
from collections import Counter
import matplotlib.pyplot as plt
from gensim.test.utils import get_tmpfile, common_texts
from gensim.models import Word2Vec
from my_log import logger
from tqdm import tqdm
import time
from lstm_crf.utils import config
import pickle
import numpy as np
from functools import reduce
from collections import defaultdict

# 分词词汇的正则
patt = re.compile("[\n_\s]+")


def word_count(files, filter_min=1, save=True):
    """
    统计词频，对于一些低频的字符什么都学不到
    :param files:
    :param filter_min:
    :return:
    """
    words = []
    tags = []
    for file in files:
        logger.info(file)
        with open(file, encoding="utf-8", mode="r") as file:
            # print(file.read())
            string = file.read()
            # 用到train data
            words.extend([w.split("/")[0] for w in patt.split(string.strip())])
            tags.extend(
                [w.split("/")[1] for w in patt.split(string.strip()) if len(w.split("/")) == 2])
            # tmp = [w.split("/") for w in re.compile("\n|_").split(file.read().strip())]
            # tags.extend(pair[1] for pair in tmp if len(words) == 2)
            # words.extend(pair[0] for pair in tmp if len(words) == 2)
            # file.closed()
            del string
    logger.info("训练数据输出的tag分布 %s" % Counter(tags))
    del tags
    counter = Counter(words)

    save_words = set()
    filter_words = set()
    for k, v in counter.items():
        if v >= filter_min:
            save_words.add(k)
        else:
            filter_words.add(k)
    logger.info("count for all(%d) save(%d) filter(%d)" % (len(counter), len(save_words), len(filter_words)))
    # print(counter)
    if save:
        with open("data/words.pkl", mode="wb") as file:
            pickle.dump(save_words, file)
            file.close()
            logger.info("save words")
    return save_words, filter_words


def word2vec(model_path, corpus, embeding_size=256, min_count=1, window=7):
    path = get_tmpfile(model_path)
    logger.info("开始训练word2vec：%s" % time.ctime())
    model = Word2Vec(sentences=corpus, size=embeding_size, min_count=min_count, window=window, workers=4, iter=10)
    logger.info("结束训练word2vec：%s" % time.ctime())
    model.save(model_path)
    # model.wv.save(wv_path)
    # print(model.wv.vocab.items())
    # model = Word2Vec.load(model_path)
    with open(model_path, encoding="utf-8", mode="w") as file:
        for word, _ in model.wv.vocab.items():
            # print(word)
            vector = [str(i) for i in model.wv[word]]
            file.write(word + " " + " ".join(vector) + "\n")


def run_word2vec(files, words, window):
    f = lambda x: x if x in words else config["unknow_word"]
    corpus = []
    for file in files:
        with open(file, encoding="utf-8", mode="r") as file:
            for line in file.readlines():
                # for train data  split("/")：可能会用到trian  data
                words_line = [w.split("/")[0] for w in patt.split(line.strip())]
                words_line_with_unk = list(map(f, words_line))
                if len(words_line) < window:
                    # print(line)
                    continue
                corpus.append([v + "_" for v in words_line_with_unk])
    logger.info("训练word2vec 的句子数量：%s" % len(corpus))
    with open("data/train_word2vec.txt", encoding="utf-8", mode="w") as file:
        string = "\n".join([" ".join(words) for words in corpus])
        file.write(string)
    # train
    # 这里频率低的已经变成了unk，min_count=1
    word2vec("data/word2vec.txt", corpus, embeding_size=256, min_count=1, window=window)


def piece2tag(string):
    piece, tag = string.split("/")
    piece_words = [v.strip() for v in piece.split("_")]
    assert len(piece_words) > 0
    result = []
    if len(piece_words) == 1:
        result.append([piece_words[0], tag + "_S"])
    else:
        result.append([piece_words[0], tag + "_B"])
        for v in piece_words[1:-1]:
            result.append([v, tag + "_M"])
        result.append([piece_words[0], tag + "_E"])
    return result


def prepare_data_lstm_crf(file, data_type):
    logger.info(file)
    assert data_type in ["train", "test", "dev"]
    file = open(file, encoding="utf-8", mode="r")
    input_seq = []
    output_seq = []
    if data_type == "test":
        for line in tqdm(file.readlines()):
            line_words = line.strip().split("_")
            input_seq.append(line_words)

    else:
        for line in tqdm(file.readlines()):
            # for line in file.readlines():
            line_words = re.findall("\d+", line)
            # input_seq.append(line_words)
            # logger.info(line)
            # train data 的piece 通过空格分割
            pieces = re.split("\s+", line.strip())
            # # logger.info(pieces)
            line_seq = []
            line_tag = []
            pieces_res = list(map(piece2tag, pieces))
            for piece_res in pieces_res:
                # logger.info(piece_res)
                for item1, item2 in piece_res:
                    line_seq.append(item1)
                    # logger.info(item1)
                    # logger.info(line_seq)
                    line_tag.append(item2)
            input_seq.append(line_seq)
            output_seq.append(line_tag)
            # logger.info(line_seq)
            # logger.info(line_tag)
            # print()
    file.close()
    # mask unk
    logger.info(input_seq[:4])

    def f1(x, y):
        return x + y

    input_seq_concat = list(reduce(f1, input_seq))
    logger.info(input_seq_concat[:500])
    logger.info("输入的词汇丰富程度%s： %d" % (data_type, len(set(input_seq_concat))))

    file = open("data/lstm_crf/%s.txt" % data_type, encoding="utf-8", mode="w")
    file_lines = ["\n".join(words) for words in input_seq]
    if data_type != "test":
        assert len(input_seq) == len(output_seq)
        file_lines = []
        for line_seqs, line_tags in zip(input_seq, output_seq):
            assert len(line_seqs) == len(line_tags)
            pair = [item1 + "\t" + item2 for item1, item2 in zip(line_seqs, line_tags)]
            file_lines.append("\n".join(pair))

    file.write("\n\n".join(file_lines))
    logger.info("%s sentences  num: %d" % (data_type, len(file_lines)))


def get_voc_dict(files, start=2):
    """
    统计词频，对于一些低频的字符什么都学不到
    :param files:
    :param filter_min:
    :return:
    """
    words = []
    for file in files:
        logger.info(file)
        with open(file, encoding="utf-8", mode="r") as file:
            # print(file.read())
            string = file.read()
            # 用到train data
            words.extend([w.split("/")[0] for w in patt.split(string.strip())])
            del string
    counter = Counter(words)
    logger.info(counter)
    word_ids = defaultdict()
    for index, word in enumerate(counter.keys()):
        word_ids[word] = index + start
    logger.info(word_ids)
    logger.info(len(word_ids))
    pickle.dump(word_ids, open("D:/projects_py/datagrand_extract_info/data/lstm_crf/train_test_word2ids.pkl", mode="wb"))
    return len(word_ids)


def get_tag_dict(file, start=1):
    """label进行标记"""
    tags = []
    with open(file, encoding="utf-8", mode="r") as file:
        line = file.readline()
        while line:
            line = line.strip()
            if line:
                word, tag = line.strip().split("\t")
                tags.append(tag)
            line = file.readline()
    counter = Counter(tags)
    logger.info(counter)
    tags_ids = defaultdict()
    for index, word in enumerate(counter.keys()):
        tags_ids[word] = index + start
    logger.info(tags_ids)
    logger.info(len(tags_ids))
    pickle.dump(tags_ids, open("D:/projects_py/datagrand_extract_info/data/lstm_crf/train_test_tag2ids.pkl", mode="wb"))
    return len(tags_ids)


if __name__ == '__main__':
    # 统计train 和test的词频
    files = ["data/train.txt"]
    word_count(files, 1, False)
    files = ["data/test.txt"]
    word_count(files, 1, False)
    # 获取train和test的词汇
    files = ["data/train.txt", "data/test.txt"]
    word2ids = get_voc_dict(files, 2)  # 0 for padding; 1 for unk
    logger.info("需要的voc  size 大小 %d" % len(word2ids))

    # files = ["data/corpus.txt", "data/train.txt"]
    # save_words, filter_words = word_count(files, 2)
    save_words = pickle.load(open("data/words.pkl", mode="rb"))

    # print(piece2tag("12266/c"))
    # print(piece2tag("17488_12266/c"))
    # print(piece2tag("17488_19311_12266/c"))

    # 将train和test准备成标准的输入模式
    prepare_data_lstm_crf("data/test.txt", "test")
    prepare_data_lstm_crf("data/train.txt", "train")
    # 必须在train准备好之后统计tag
    get_tag_dict("data/lstm_crf/train.txt", 1)  # 0 for padding
    files = ["data/corpus.txt", "data/train.txt", "data/test.txt"]
    run_word2vec(files, words=save_words, window=6)
