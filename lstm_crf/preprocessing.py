# -*- encoding: utf-8 -*-
"""
    预处理
"""
import sys
import yaml
import pickle
import codecs
import numpy as np
from collections import defaultdict
from utils import create_dictionary, load_embed_from_txt
from my_log import logger

"""
根据设定的特征，生成voc，并结合embeding 生成matrix，input
model:
trian_path:
"""


def build_vocabulary(path_data, path_vocs_dict, min_counts_dict, columns,
                     sequence_len_pt=98, use_char_featrue=False, word_len_pt=98):
    """
    构建字典
    Args:
        path_data: str, 数据路径
        path_vocs_dict: dict, 字典存放路径
        min_counts_dict: dict, item最少出现次数
        columns: list of str, 每一列的名称
        sequence_len_pt: int，句子长度百分位
        use_char_featrue: bool，是否使用字符特征(针对英文， 每个词汇的构成也可以成为一个输入信号)
        word_len_pt: int，单词长度百分 位
    Returns:
        voc_size_1, voc_size_2, ...: int
        sequence_length: 序列最大长度
    """
    logger.info("对特征和输出的label建立vocabulary, data:" + path_data)
    logger.info("信号的特征：" + "\t".join(columns))
    logger.info("stat features(char) and label frequency...")
    file_data = codecs.open(path_data, 'r', encoding='utf-8')
    line = file_data.readline()
    sequence_length_list = []  # 句子长度统计集合，用于计算设定的分位数句子长度
    # 计数items
    feature_item_dict_list = []
    for i in range(len(columns)):
        # 每个特征的信号频数统计结果的字典初始化， 默认值为0
        feature_item_dict_list.append(defaultdict(int))
    # char feature
    if use_char_featrue:
        char_dict = defaultdict(int)
        word_length_list = []  # 单词长度
    sequence_length = 0  # 统计句子包含信号的长度
    sentence_count = 0  # 统计句子数
    while line:
        line = line.rstrip()
        if not line:
            # 分句空行位置
            sentence_count += 1
            sys.stdout.write('当前处理句子数: %d\r' % sentence_count)
            sys.stdout.flush()
            line = file_data.readline()
            sequence_length_list.append(sequence_length)  # 句子的长度
            sequence_length = 0  # 计算下一句
            continue
        items = line.split('\t')  # 以制表符分开
        sequence_length += 1
        # print(items)
        for i in range(len(columns) - 1):  # 前n-1个特征， label不需要统计？？
            feature_item_dict_list[i][items[i]] += 1  # 统计的是每个特征的词频
        feature_item_dict_list[-1][items[-1]] += 1  # label
        # char feature
        if use_char_featrue:  # 如果是英文， 统计每个字符的频数
            for c in items[0]:  # 第一列为英文词汇，统计词汇每个char
                char_dict[c] += 1
            word_length_list.append(len(items[0]))
        # next loop
        line = file_data.readline()
    file_data.close()
    # print("stat count: ", feature_item_dict_list)  # f + label
    # last instance
    if sequence_length != 0:
        sentence_count += 1
        sequence_length_list.append(sequence_length)
        sys.stdout.write('当前处理句子数: %d\r' % sentence_count)
        sys.stdout.flush()

    logger.info("build features(char) and label voc...")
    voc_sizes = []  # 每个特征的词典的size
    if use_char_featrue:
        # char feature 编号
        logger.info("create dictionary: english char")
        size = create_dictionary(
            char_dict,
            path_vocs_dict['char'],
            start=2,  # 填充，未登录 分别占位 0， 1 index
            sort=True,
            min_count=min_counts_dict['char'],
            overwrite=True
        )
        voc_sizes.append(size)
        logger.info('voc: char, size: %d' % size)

    for i, name in enumerate(columns):
        logger.info("create dictionary: %s" % name)
        # 特征的start=2， label start=1 这样导致label的种类数量会加1（序列长度不够部分用0填充）
        start = 1 if i == len(columns) - 1 else 2
        size = create_dictionary(
            feature_item_dict_list[i],
            path_vocs_dict[name],
            start=start,
            sort=True,
            min_count=min_counts_dict[name],
            overwrite=True
        )
        voc_sizes.append(size)
        logger.info('voc: %s, size: %d' % (path_vocs_dict[name], size))

    logger.info('句子长度分布:')
    sentence_length = -1
    option_len_pt = [90, 95, 98, 100]
    if sequence_len_pt not in option_len_pt:
        option_len_pt.append(sequence_len_pt)
    for per in sorted(option_len_pt):
        tmp = int(np.percentile(sequence_length_list, per))
        # 句子长度的分位数计算
        if per == sequence_len_pt:
            sentence_length = tmp  # 最终选择输入信号句子固定的长度
            logger.info('%3d percentile: %d (default)' % (per, tmp))
        else:
            logger.info('%3d percentile: %d' % (per, tmp))

    if use_char_featrue:
        logger.info('单词长度分布:')
        word_length = -1
        option_len_pt = [90, 95, 98, 100]
        if word_len_pt not in option_len_pt:
            option_len_pt.append(word_len_pt)
        for per in sorted(option_len_pt):
            tmp = int(np.percentile(word_length_list, per))
            if per == word_len_pt:
                word_length = tmp  # 最终选择输出字符串的最大长度
                logger.info('%3d percentile: %d (default)' % (per, tmp))
            else:
                logger.info('%3d percentile: %d' % (per, tmp))

    lengths = [sentence_length]
    if use_char_featrue:
        lengths.append(word_length)
    logger.info("build_vocabulary completely!")
    return voc_sizes, lengths  # 每个特诊的去重和按频数过滤后的量， 句子长度[英语字符的长度]


def main():
    logger.info('preprocessing...')
    useable = []
    # 加载配置文件
    with open('./config.yml', encoding="utf-8") as file_config:
        config = yaml.load(file_config)

    # 构建字典(同时获取词表size，序列最大长度)， f1 f2 label 名称固定不能更改
    # 输入特征[f1]或者[f1, f2], f1: 汉字或者英语词汇，f2：词性， 加上一个为预测label
    columns = config['model_params']['feature_names'] + ['label']
    min_counts_dict, path_vocs_dict = defaultdict(int), dict()  # 用来过滤的最小频数，整理编号词汇的保存路径
    feature_names = config['model_params']['feature_names']  # 输入信号的特征
    logger.info("feature_names: " + str(feature_names))
    for feature_name in feature_names:
        min_counts_dict[feature_name] = config['data_params']['voc_params'][feature_name]['min_count']
        path_vocs_dict[feature_name] = config['data_params']['voc_params'][feature_name]['path']
    # label的编号保存路径
    path_vocs_dict['label'] = config['data_params']['voc_params']['label']['path']
    logger.info("min_count: " + str(min_counts_dict))
    logger.info(path_vocs_dict)

    # char feature  char 命名也是固定不能更改
    min_counts_dict['char'] = config['data_params']['voc_params']['char']['min_count']
    path_vocs_dict['char'] = config['data_params']['voc_params']['char']['path']

    sequence_len_pt = config['model_params']['sequence_len_pt']  # 句子长度覆盖分位数
    use_char_feature = config['model_params']['use_char_feature']  # 是否属于英文本文本并采用字符信息
    word_len_pt = config['model_params']['word_len_pt']  # 英文文本每个字符的长度控制
    # 将输入用到的输入和输出特征建立遍序号字典保存
    voc_sizes, lengths = build_vocabulary(
        path_data=config['data_params']['path_train'],
        columns=columns,
        min_counts_dict=min_counts_dict,
        path_vocs_dict=path_vocs_dict,
        sequence_len_pt=sequence_len_pt,
        use_char_featrue=use_char_feature,
        word_len_pt=word_len_pt
    )
    # logger.info(voc_sizes)
    if not use_char_feature:
        sequence_length = lengths[0]  # 预测句子长度
    else:
        sequence_length, word_length = lengths[:]  # 或者英语用到字符的信号

    # 构建embedding表， 对每个输入的特征进行embed
    logger.info("get feature pre_train matrix...")  # 模型自带一个word2vec层， word2vec的初始化权重
    feature_dim_dict = dict()  # 存储每个feature的dim
    for i, feature_name in enumerate(feature_names):
        logger.info("feature: " + feature_name)
        # embed size
        path_pre_train = config['model_params']['embed_params'][feature_name]['path_pre_train']  # embed 结果保存位置
        if not path_pre_train:
            # 检查嵌入维度初始化权重是否存在,如果不存在为啥还需要给定一个默认维度值, 改为None， 该特征不可用
            if i == 0:
                feature_dim_dict[feature_name] = None
            else:
                feature_dim_dict[feature_name] = None
            continue
        useable.append(feature_name)
        config['model_params']['embed_params'][feature_name]['path'] = "./Res/embed/%s_embed.mat.pkl" % feature_name
        path_voc = config['data_params']['voc_params'][feature_name]['path']  # 前面的特征词典编号文件位置
        with open(path_voc, 'rb') as file_r:  # 二进制打开文件
            voc = pickle.load(file_r)
        # print("编号词典：", voc)
        logger.info("将构建的voc，与训练好的embedding结合整理出word2vec的初始化矩阵: " + feature_name)
        embedding_dict, vec_dim = load_embed_from_txt(path_pre_train)  # 读取嵌入词汇模型
        feature_dim_dict[feature_name] = vec_dim  # 每个特征的嵌入维度
        embedding_matrix = np.zeros((len(voc.keys()) + 2, vec_dim), dtype='float32')  # 为啥加 2 两个全为0 的在头部两行即index=0，1
        for item in voc:
            if item in embedding_dict:
                embedding_matrix[voc[item], :] = embedding_dict[item]
            else:
                # voc里面的词汇可能在pre train embed里面没有
                embedding_matrix[voc[item], :] = np.random.uniform(-0.25, 0.25, size=vec_dim)  # embed钟未登录词的处理
        with open(config['model_params']['embed_params'][feature_name]['path'], 'wb') as file_w:
            pickle.dump(embedding_matrix, file_w)
        # print(embedding_matrix)
    # 修改config中各个特征的shape，embedding大小默认为[64, 32, 32, ...]
    if use_char_feature:
        char_voc_size = voc_sizes.pop(0)
    label_size = voc_sizes[-1]
    voc_sizes = voc_sizes[:-1]  # 仅仅包含f
    # 修改nb_classes
    config['model_params']['nb_classes'] = label_size  # 实际的分类数量
    # 修改embedding表的shape
    for i, feature_name in enumerate(feature_names):
        config['model_params']['embed_params'][feature_name]['shape'] = [voc_sizes[i], feature_dim_dict[feature_name]]  # 保存特征集合大小 和 enbed size
    # 修改char表的embedding
    if use_char_feature:
        # 默认16维，根据任务调整
        # 并且没有生成char mat： char_embed.pkl 这是事先给出的
        config['model_params']['embed_params']['char']['shape'] = [char_voc_size, 16]  # 固定到16 dim
        config['model_params']['word_length'] = word_length
    # 修改句子长度
    config['model_params']['sequence_length'] = sequence_length
    config['model_params']['feature_names'] = useable
    # 训练多个标注模型， 根据model 来保存训练模型
    config["model_params"]["path_model"] = "./Model/" + config["model"] + "/best_model"
    # 根据数据情概，更新配置文件
    with codecs.open('./config.yml', 'w', encoding='utf-8') as file_w:
        yaml.dump(config, file_w)
    logger.info('preprocessing successfully!')


if __name__ == '__main__':
    main()
