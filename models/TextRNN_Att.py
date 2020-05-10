# coding:utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.ConfigBase import ConfigBase


class Config(object):
    """配置参数"""

    def __init__(self, dataset, embedding):
        super(Config, self).__init__(dataset, embedding)
        self.model_name = 'TextRNN_Att'

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 10  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度： 小于该值则填充，大于则截断
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.hidden_size = 256  # lstm 隐藏层
        self.num_layers = 2   # lstm层数


class Model(nn.Module):
    """"""

    def __init__(self, config):
        super(Model, self).__init__()

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(
                config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.text_length = config.batch_size
        self.lstm = nn.LSTM(config.embed,
                            config.hidden_size,
                            config.num_layers,
                            bidirectional=True,
                            batch_first=True,
                            dropout=config.dropout)

        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        # self.tanh = nn.Tanh()
        self.w = nn.Parameter(torch.Tensor(config.hidden_size * 2))
        self.dropout = nn.Dropout(config.dropout)
        # self.W_w = nn.Parameter(torch.Tensor(
        #     config.hidden_size * 2, config.hidden_size * 2))

    def forward(self, x):
        embedded = self.dropout(self.embedding(x[0]))
        H, _ = self.lstm(embedded)

        M = torch.tanh(H)

        attention_weights = F.softmax(torch.matmul(M, self.w),
                                      dim=1).unsqueeze(-1)
        out = H * attention_weights
        out = torch.sum(out, 1)
        out = F.relu(out)
        out = self.fc(out)
        return out
