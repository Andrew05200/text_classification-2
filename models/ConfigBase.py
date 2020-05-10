# coding: UTF-8
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConfigBase(object):
    """配置参数"""

    def __init__(self, dataset, embedding=None):
        self.model_name = ''
        self.train_path = './data/THUCNews/train.txt'  # 训练集
        self.dev_path = './data/THUCNews/dev.txt'  # 验证集
        self.test_path = './data/THUCNews/test.txt'  # 测试集
        self.vocab_path = './data/THUCNews/vocab.pkl'  # 词表
        self.save_path = './data/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果
        self.log_path = './data/log/' + self.model_name
        if not os.path.exists("./data/saved_dict/"):
            os.mkdir("./data/saved_dict")
        if not os.path.exists("./data/log/"):
            os.mkdir("./data/log")
        self.class_list = [
            x.strip() for x in open('./data/THUCNews/class.txt').readlines()
        ]  # 类别名单
        self.embedding_pretrained = torch.tensor(
            np.load('./data/THUCNews/embedding_SougouNews.npz')["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')  # 设备
            
        print("cuda : ", torch.cuda.is_available())
