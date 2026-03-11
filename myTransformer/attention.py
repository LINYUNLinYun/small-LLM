from torch import nn
import torch.nn.functional as F
import torch
import math

def attention(query, key, value, dropout = None):
    # 获取键向量的维度 query: (BS, num_heads, seqlen, d_embd = d_k)
    d_k = query.size(-1)
    # 计算QK矩阵乘积 (BS num_heads seqlen d_k) * (BS num_heads vacab_num d_k) = (BS num_heads seqlen vocab_num)
    scores = torch.matmul(query, key.transpose(-2,-1))/math.sqrt(d_k)
    # 对最后一维，即在列上滑动，提取每个query行的所有列做
    p_attn = scores.softmax(dim=-1)
    # 采样
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn