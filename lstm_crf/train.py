#!/usr/bin/env python
# -*- encoding: utf-8 -*-

"""
    训练NER模型
"""
import yaml
import pickle
from load_data import load_vocs, init_data
from model import SequenceLabelingModel
from my_log import logger


def main():
    # 加载配置文件
    with open('./config.yml') as file_config:
        config = yaml.load(file_config)

    feature_names = config['model_params']['feature_names']
    logger.info(feature_names)
    use_char_feature = config['model_params']['use_char_feature']
    logger.info(use_char_feature)
    # 初始化embedding shape, dropouts, 预训练的embedding也在这里初始化)
    feature_weight_shape_dict = dict()
    feature_weight_dropout_dict = dict()
    feature_init_weight_dict = dict()
    for feature_name in feature_names:
        feature_weight_shape_dict[feature_name] = config['model_params']['embed_params'][feature_name]['shape']
        feature_weight_dropout_dict[feature_name] = config['model_params']['embed_params'][feature_name]['dropout_rate']
        # embeding mat, 比voc多了两行， 因为voc从2开始编序， 0， 1行用0填充
        path_pre_train = config['model_params']['embed_params'][feature_name]['path']  # 词嵌矩阵位置
        # logger.info("%s init mat path: %s" % (feature_name, path_pre_train))
        with open(path_pre_train, 'rb') as file_r:
            feature_init_weight_dict[feature_name] = pickle.load(file_r)
    logger.info(feature_weight_dropout_dict)
    logger.info(feature_weight_shape_dict)
    logger.info(feature_init_weight_dict)

    # char embedding shape
    if use_char_feature:
        # 暂时不考虑
        feature_weight_shape_dict['char'] = config['model_params']['embed_params']['char']['shape']
        conv_filter_len_list = config['model_params']['conv_filter_len_list']
        conv_filter_size_list = config['model_params']['conv_filter_size_list']
    else:
        # 利用卷集层来提取char的信息
        conv_filter_len_list = None
        conv_filter_size_list = None

    # 加载vocs
    path_vocs = []
    if use_char_feature:
        path_vocs.append(config['data_params']['voc_params']['char']['path'])  # vocs用于将文本数字序列化
    for feature_name in feature_names:
        path_vocs.append(config['data_params']['voc_params'][feature_name]['path'])
    path_vocs.append(config['data_params']['voc_params']['label']['path'])
    vocs = load_vocs(path_vocs)

    # 加载训练数据
    sep_str = config['data_params']['sep']
    assert sep_str in ['table', 'space']  # 数据的分隔方式
    sep = '\t' if sep_str == 'table' else ' '
    max_len = config['model_params']['sequence_length']
    word_len = config['model_params']['word_length']

    # 通过voc 将input f1 和输出 label 数字序列化 得到训练的输入和输出
    # data_dict = None
    data_dict = init_data(
        path=config['data_params']['path_train'],
        feature_names=feature_names,
        sep=sep,
        vocs=vocs,
        max_len=max_len,
        model='train',
        use_char_feature=use_char_feature,
        word_len=word_len
    )
    logger.info(data_dict)  # 每个特征序列化后的数据
    # 训练模型
    model = SequenceLabelingModel(
        sequence_length=config['model_params']['sequence_length'],  # 句子被固定长度
        nb_classes=config['model_params']['nb_classes'],
        nb_hidden=config['model_params']['bilstm_params']['num_units'],
        num_layers=config['model_params']['bilstm_params']['num_layers'],
        rnn_dropout=config['model_params']['bilstm_params']['rnn_dropout'],
        feature_weight_shape_dict=feature_weight_shape_dict,
        feature_init_weight_dict=feature_init_weight_dict,
        feature_weight_dropout_dict=feature_weight_dropout_dict,
        dropout_rate=config['model_params']['dropout_rate'],
        nb_epoch=config['model_params']['nb_epoch'], feature_names=feature_names,
        batch_size=config['model_params']['batch_size'],
        train_max_patience=config['model_params']['max_patience'],
        use_crf=config['model_params']['use_crf'],
        l2_rate=config['model_params']['l2_rate'],
        rnn_unit=config['model_params']['rnn_unit'],
        learning_rate=config['model_params']['learning_rate'],
        clip=config['model_params']['clip'],
        use_char_feature=use_char_feature,
        conv_filter_size_list=conv_filter_size_list,
        conv_filter_len_list=conv_filter_len_list,
        cnn_dropout_rate=config['model_params']['conv_dropout'],
        word_length=word_len,
        path_model=config['model_params']['path_model'],
        last_train_sess_path=None,   # 为了加快训练的速度我们继续载入前面训练的参数
        transfer=False
    )   # 是否对前面载入的参数进行迁移学习，True的话就重置LSTM的输出层

    model.fit(data_dict=data_dict, dev_size=config['model_params']['dev_size'])
    """
    # 上次训练的参数结果：
    NER_ins: train loss: 0.918179, dev loss: 0.677157, l2 loss: 0.025351
    """


if __name__ == '__main__':
    main()
