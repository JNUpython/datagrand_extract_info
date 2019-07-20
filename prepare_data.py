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


# 分词词汇的正则
patt = re.compile("[\n_\s]+")

def word_count(files, filter_min=1):
    """
    统计词频，对于一些低频的字符什么都学不到
    :param files:
    :param filter_min:
    :return:
    """
    words = []
    tags = []
    for file in files:
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
    logger.info("count for all(%d) save(%d) filter(%d)" % (len(counter), len(filter_words), len(save_words)))
    # print(counter)
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


def run_word2vec(files, unk_words, window):
    f = lambda x: x if x not in unk_words else "unk"
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
            result.append([piece_words[0], tag + "_M"])
        result.append([piece_words[0], tag + "_E"])
    return result


def prepare_data_lstm_crf(file, data_type, unk_words=set()):
    f = lambda x: x if x not in unk_words else "unk"
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
            # train data 的piece 通过空格分割
            pieces = re.split("\s+", line.strip())
            line_input_seq = []
            line_output_seq = []
            for piece_res in map(piece2tag, pieces):
                for item1, item2 in piece_res:
                    line_input_seq.append(item1)

                    line_output_seq.append(item2)
            input_seq.append(line_input_seq)
            output_seq.append(line_output_seq)
    file.close()
    # mask unk
    input_seq = [list(map(f, line)) for line in input_seq]
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


if __name__ == '__main__':
    files = ["data/corpus.txt", "data/train.txt"]
    save_words, filter_words = word_count(files, 2)
    # print(piece2tag("12266/c"))
    # print(piece2tag("17488_12266/c"))
    # print(piece2tag("17488_19311_12266/c"))
    prepare_data_lstm_crf("data/test.txt", "test", filter_words)
    prepare_data_lstm_crf("data/train.txt", "train", filter_words)
    files = ["data/corpus.txt", "data/train.txt", "data/test.txt"]
    run_word2vec(files, unk_words=filter_words, window=6)
