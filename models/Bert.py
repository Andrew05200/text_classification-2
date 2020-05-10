# coding:utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from pytorch_transformers.modeling_bert import BertModel, BertPreTrainedModel, BertConfig
# from pytorch_transformers.tokenization_bert import BertTokenizer
from transformers import BertPreTrainedModel, BertModel
from transformers import BertTokenizer
from models.ConfigBase import ConfigBase


class Config(ConfigBase):
    def __init__(self, dataset):
        super(Config, self).__init__(dataset)
        self.model_name = 'Bert'

        self.require_improvement = 1000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数   
        self.n_vocab = 0  # 词表大小，在运行时赋值
        self.num_epochs = 10  # epoch数
        self.batch_size = 128  # mini-batch大小
        self.pad_size = 32  # 每句话处理成的长度： 小于该值则填充，大于则截断
        self.learning_rate = 5e-5  # 学习率
        self.bert_path = "./bert-base-chinese"

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768


# class Model(nn.Module):
class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        
        self.bert = BertModel.from_pretrained(config.bert_path)

        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        context = x[0]
        mask = x[2]
        _, pooled = self.bert(context, attention_mask=mask)
        out = self.fc(pooled)
        return out
