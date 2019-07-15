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


def word_count(files, filter_min=1):
    """
    统计词频，对于一些低频的字符什么都学不到
    :param files:
    :param filter_min:
    :return:
    """
    words = []
    for file in files:
        with open(file, encoding="utf-8", mode="r") as file:
            # print(file.read())
            words.extend([w.split("/")[0] for w in re.compile("\n|_").split(file.read().strip())])
            # file.closed()
    counter = Counter(words)
    save_num = 0
    filter_num = 0
    for k, v in counter.items():
        if v > filter_min:
            save_num += 1
        else:
            filter_num += 1
    print(len(counter), filter_num, save_num)
    print(counter)


def word2vec(model_path, corpus, embeding_size=256, min_count=2, window=5):
    path = get_tmpfile(model_path)
    model = Word2Vec(sentences=corpus, size=embeding_size, min_count=min_count, window=window, workers=2, iter=5, sg=1)
    model.save(model_path)
    # model.wv.save(wv_path)
    # print(model.wv.vocab.items())
    # model = Word2Vec.load(model_path)
    with open(model_path, encoding="utf-8", mode="w") as file:
        for word, _ in model.wv.vocab.items():
            print(word)
            vector = [str(i) for i in model.wv[word]]
            file.write(word + " " + " ".join(vector) + "\n")


if __name__ == '__main__':
    files = ["data/corpus.txt", "data/train.txt"]
    # word_count(files, 3)
    corpus = []
    for file in files:
        with open(file, encoding="utf-8", mode="r") as file:
            for line in file.readlines():
                words_line = [w.split("/")[0] for w in line.strip().split("_")]
                if len(words_line) < 5:
                    # print(line)
                    continue
                corpus.append(words_line)
    print(len(corpus))
    with open("data/train_word2vec.txt", encoding="utf-8", mode="w") as file:
        string = "\n".join([" ".join(words) for words in corpus])
        file.write(string)
    word2vec("data/word2vec.txt", corpus)
