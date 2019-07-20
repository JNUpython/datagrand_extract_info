#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
    加载数据
"""
import sys
import codecs
import pickle
import numpy as np
from utils import map_item2id
from my_log import logger


def load_vocs(paths):
    """
    加载vocs, 每个特征的编号字典
    Args:
        paths: list of str, voc路径
    Returns:
        vocs: list of dict
    """
    vocs = []
    for path in paths:
        with open(path, 'rb') as file_r:
            vocs.append(pickle.load(file_r))
    return vocs


def load_lookup_tables(paths):
    """
    加载lookup tables， 每个特征的embedding
    Args:
        paths: list of str, emb路径
    Returns:
        lookup_tables: list of dict
    """
    lookup_tables = []
    for path in paths:
        with open(path, 'rb', encoding='utf-8') as file_r:
            lookup_tables.append(pickle.load(file_r))
    return lookup_tables


def init_data(path,
              feature_names,
              vocs,
              max_len,
              model='train',
              use_char_feature=False,
              word_len=None,
              sep='\t'
              ):
    """
    加载数据(待优化，目前是一次性加载整个数据集)， 每个特征按序列化按字典保存
    Args:
        path: str, 数据路径
        feature_names: list of str, 特征名称
        vocs: list of dict
        max_len: int, 句子最大长度
        model: str, in ('train', 'test')
        use_char_feature: bool，是否使用char特征
        word_len: None or int，单词最大长度
        sep: str, 特征之间的分割符, default is '\t'
    Returns:
        data_dict: dict
    """
    logger.info("initialize data, model: " + model)
    logger.info("载入数据：" + path)
    assert model in ('train', 'test')
    file_r = open(path, 'r', encoding='utf-8')
    sentences = file_r.read().strip().split('\n\n')  # 分成句子
    # print(sentences)
    sentence_count = len(sentences)
    logger.info("句子数量：%d" % sentence_count)
    feature_count = len(feature_names)
    logger.info("num for features: %d" % feature_count)
    data_dict = dict()  # 用字典保存每个特征mat化的信息
    for feature_name in feature_names:
        # 先定义一个默认的都为0的初始化mat， 将字符转化为对应的编码
        data_dict[feature_name] = np.zeros((sentence_count, max_len), dtype='int32')  # 句子数量 x 最大句子长度
    # char feature
    if use_char_feature:
        # 如果是英文 就可以 将每个单词拆成一个个字符， word_len 组成一个信号最大字符长度， 汉字就是一个个字组成无法进行进一步的拆分
        data_dict['char'] = np.zeros((sentence_count, max_len, word_len), dtype='int32')
        char_voc = vocs.pop(0)  # 用完就删除
    if model == 'train':
        # 如果数据是用来训练, 就需要定义出书label数据
        data_dict['label'] = np.zeros((sentence_count, max_len), dtype='int32')

    # 按特征类别整理每个句子的分类
    for index, sentence in enumerate(sentences):
        # print("序列化句子：", repr(sentence))
        items = sentence.split('\n')  # 每个句子分词一个个信号 以及label（测试数据就不需要）
        one_instance_items = []
        [one_instance_items.append([]) for _ in range(feature_count + 1)]  # 多个特征 + label
        for item in items:
            feature_tokens = item.strip().split(sep)
            for j in range(feature_count):
                # 嵌n个input特征
                one_instance_items[j].append(feature_tokens[j])
            if model == 'train':
                # 最后一个output
                one_instance_items[-1].append(feature_tokens[-1])
        for i in range(len(feature_names)):
            # 特征序列化：映射成对应的id
            data_dict[feature_names[i]][index, :] = map_item2id(
                one_instance_items[i],
                vocs[i],
                max_len
            )  # 输入特征每个序列化

        if use_char_feature:
            for i, word in enumerate(one_instance_items[0]):  # 默认第一特诊为包含字符集的特征
                if i >= max_len:
                    break
                data_dict['char'][index][i, :] = map_item2id(
                    word,
                    char_voc,
                    word_len
                )  # 这样的特征为一个二维的类似于图像 num_batch * num_word * word_max_len

        if model == 'train':
            data_dict['label'][index, :] = map_item2id(
                one_instance_items[-1],
                vocs[-1],
                max_len
            )
        sys.stdout.write('loading data: %d\r' % index)
        sys.stdout.flush()
    file_r.close()
    logger.info("initialize completely")
    return data_dict
