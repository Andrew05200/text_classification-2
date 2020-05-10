# coding: UTF-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.ConfigBase import ConfigBase

class Config(ConfigBase):
    """配置参数"""

    def __init__(self, dataset, embedding):
        super(Config, self).__init__(dataset, embedding)
        self.model_name = 'TextCNN'
        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 20  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度： 小于该值则填充，大于则截断
        self.learning_rate = 1e-3  # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)  # 卷积核尺寸
        self.num_filters = 256  # 卷积核数量(channels数)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        Vocab = config.n_vocab
        Dim = config.embed
        cla = config.num_classes
        Ci = 1  # 输入channel数
        kernal_nums = config.num_filters
        Ks = config.filter_sizes

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(Vocab, Dim,
                                          padding_idx=Vocab - 1)
        self.convs = nn.ModuleList([
            nn.Conv2d(Ci, kernal_nums, (k, Dim))
            for k in Ks
        ])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(kernal_nums * len(Ks), cla)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv)
                         for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out
