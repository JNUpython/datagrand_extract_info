# 达观信息抽取比赛项目

idea：
1. 统计训练和测试数据的句子长度： max_sentence_length 取95%分位数
2. 统计下corpus的词频freq 以及 voc_size， 看看是否定频词汇比较多，比较多就filter低频词汇，统一标记为unk
3. 统计下 目标tag 前后词汇
4. model archive：
    - char-embed + bilstm + crf
    - 按照123的insight确定voc_size 和 max_sentence_length 然后训练word2vec
    - 分词考虑wordPiece 降低统计voc_size 大小，但是不能把目标tag wordPiece 处理掉，统计是否能降低one-hot维度，即voc_size 大小
    - 训练word2vec embedding-size 256或者512