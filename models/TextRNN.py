# coding: UTF-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ConfigBase import ConfigBase


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        super(Config, self).__init__(dataset, embedding)
        self.model_name = 'TextRNN'

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度： 小于该值则填充，大于则截断
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度

        self.hidden_size = 128  # lstm 隐藏层数量
        self.num_layers = 2  # lstm 层数


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab,
                                          config.embeding,
                                          padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed,
                            config.hidden_size,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        # x, _ = x
        # [batch_size, seq_len, embeding]=[128, 32, 300]
        out = self.embedding(x[0])
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out
