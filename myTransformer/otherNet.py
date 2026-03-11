import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from modelArgs import ModelArgs

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()

        # 输入到隐层 线性层
        self.w1 = nn.Linear(dim, hidden_dim,bias=False)
        # 隐层到输出 线性层
        self.w2 = nn.Linear(hidden_dim, dim, bias= False)
        # dropout
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        return self.dropout(self.w2(F.relu(self.w1(x))))
    
class LayerNorm(nn.Module):
    def __init__(self, features, eps = 1e-6):
        super().__init__()
        # features应该是个标量
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))

        self.eps = eps
        
    def forward(self, x):
        # 在统计每个样本所有维度的值，求均值和方差
        # 对最后一个维度做均值 [bs, seqlen, embd_dim] -> [bsz, max_len, 1]
        mean = x.mean(-1, keepdim = True)
        std = x.std(-1, keepdim = True)

        return self.a*(x - mean)/(std + self.eps) + self.b

class PositionalEncoding(nn.Module):
    '''位置编码模块'''
    def __init__(self, args):
        super().__init__()

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算 theta 将指数分母化为指对数函数
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)    # 在第0维加一个维度方便和嵌入向量做广播相加
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x
