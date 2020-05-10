# coding:utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import copy
from models.ConfigBase import ConfigBase

'''
https://juejin.im/post/5b9f1af0e51d450e425eb32d
'''


class Config(object):
    def __init__(self, dataset, embedding):
        super(Config, self).__init__(dataset, embedding)
        self.model_name = 'Transformer'

        self.dropout = 0.5                                              # 随机失活
        # 若超过1000batch效果还没提升，则提前结束训练
        self.require_improvement = 2000
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        # 每句话处理成的长度(短填长切)
        self.pad_size = 32
        self.learning_rate = 1e-3                                     # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.dim_model = 300
        self.hidden = 1024
        self.last_hidden = 512
        self.num_head = 5
        self.num_encoder = 2


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(
                config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(
                config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)

        self.postion_embedding = Positional_Encoding(
            config.embed, config.pad_size, config.dropout, config.device)
        self.encoder = Encoder(
            config.dim_model, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            # Encoder(config.dim_model, config.num_head, config.hidden, config.dropout)
            for _ in range(config.num_encoder)])

        self.fc = nn.Linear(
            config.pad_size * config.dim_model, config.num_classes)
        # self.fc2 = nn.Linear(config.last_hidden, config.num_classes)
        # self.fc1 = nn.Linear(config.dim_model, config.num_classes)

    def forward(self, x):
        out = self.embedding(x[0])
        out = self.postion_embedding(out)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc(out)
        return out


class Encoder(nn.Module):
    def __init__(self, dim_model, num_head, hidden, dropout):
        super(Encoder, self).__init__()
        self.attention = Multi_Head_Attention(dim_model, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(
            dim_model, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class Positional_Encoding(nn.Module):
    '''
    位置编码
    '''

    def __init__(self, embed, pad_size, dropout, device):
        super(Positional_Encoding, self).__init__()
        self.device = device
        self.pe = torch.tensor([[pos / (10000.0 ** (i // 2 * 2.0 / embed))
                                 for i in range(embed)] for pos in range(pad_size)])
        # 偶数列使用sin，奇数列使用cos
        self.pe[:, 0::2] = np.sin(self.pe[:, 0::2])
        self.pe[:, 1::2] = np.cos(self.pe[:, 1::2])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = x + nn.Parameter(self.pe,
                               requires_grad=False).to(self.device)
        out = self.dropout(out)
        return out


class Scaled_Dot_Product_Attention(nn.Module):
    """
        self attention 基本单元
    """

    def __init__(self, attention_dropout=0.0):
        super(Scaled_Dot_Product_Attention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, Q, K, V, scale=None, attention_mask=None):
        """
            Q、K、V均来自同一个输入
            Q: [batch_size, len_Q, dim_Q]
            K: [batch_size, len_K, dim_K]
            V: [batch_size, len_V, dim_V]
            scale: 缩放因子 论文为根号dim_K
        Return:
            self-attention后的张量，以及attention张量

        encoder的self-attention中，Q、K、V都来自同一个地方（相等），他们是上一层encoder的输出。
        对于第一层encoder，它们就是word embedding和positional encoding相加得到的输入。

        decoder的self-attention中，Q、K、V都来自于同一个地方（相等），它们是上一层decoder的输出。
        对于第一层decoder，它们就是word embedding和positional encoding相加得到的输入。
        在encoder-decoder attention中，Q来自于decoder的上一层的输出，K和V来自于encoder的输出，K和V是一样的。
        Q、K、V三者的维度一样，即 d_q=d_k=d_v
        """
        attention = torch.matmul(Q, K.permute(0, 2, 1))

        if scale:
            attention = attention * scale
        # if attention_mask:
        #     attention = attention.mask_fill(attention_mask, -np.inf)

        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, V)
        return context


class Multi_Head_Attention(nn.Module):
    '''
    Q、K、V通过一个线性映射后，分层h份， 对每一份进行 Scaled_Dot_Product_Attention效果更好。
    然后，将各个部分的结果合并，再经过线性映射，得到最终结果
    h就是heads的数量。
    注意：这里的h份，是在dk, dp, dv的维度上进行切分，
    '''

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(Multi_Head_Attention, self).__init__()
        self.num_heads = num_heads
        self.dim_per_head = model_dim // num_heads

        self.liner_q = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.liner_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.liner_v = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.attention = Scaled_Dot_Product_Attention(dropout)
        # self.attention = Scaled_Dot_Product_Attention()
        self.liner_final = nn.Linear(
            self.num_heads * self.dim_per_head, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x, attention_mask=None):
        batch_size = x.size(0)
        Q = self.liner_q(x)
        K = self.liner_k(x)
        V = self.liner_v(x)

        Q = Q.view(batch_size * self.num_heads, -1, self.dim_per_head)
        K = K.view(batch_size * self.num_heads, -1, self.dim_per_head)
        V = V.view(batch_size * self.num_heads, -1, self.dim_per_head)

        if attention_mask:
            attention_mask = attention_mask.repeat(self.num_heads, 1, 1)

        # scale = (K.size(-1) // self.num_heads) ** -0.5  # 缩放因子
        scale = (K.size(-1)) ** -0.5  # 缩放因子
        context = self.attention(Q, K, V, scale)

        context = context.view(
            batch_size, -1, self.dim_per_head * self.num_heads)

        out = self.liner_final(context)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super(Position_wise_Feed_Forward, self).__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out
