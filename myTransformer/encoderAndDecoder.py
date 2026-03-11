import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from modelArgs import ModelArgs
from multiHeadAttention import MultiHeadAttention
from otherNet import LayerNorm, MLP

class EncoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 注意力层归一化
        self.attn_norm = LayerNorm(args.n_embd)
        # 多头注意力
        self.mhattn = MultiHeadAttention(args)
        # 前馈层归一化 这里维度应该为dim因为mha输出的是dim
        self.fnn_norm = LayerNorm(args.dim)
        # 前馈神经网络
        self.fnn = MLP(args.dim,args.dim,args.dropout)

    def forward(self, x):
        norm_x = self.attn_norm(x)
        # 自注意力 这里还要做pre norm的残差连接
        h = x + self.mhattn.forward(norm_x, norm_x, norm_x)
        # 在fnn前层归一化
        output  = h + self.fnn.forward(self.fnn_norm(h))
        # fnn
        return output
    
class Encoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layers)])
        # norm
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        # 经过n层encoder layers 归一化出去
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attn_norm_1 = LayerNorm(args.n_embd)
        self.mask_attn = MultiHeadAttention(args, is_causal = True)
        self.attn_norm_2 = LayerNorm(args.n_embd)
        self.attn = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.dim)
        self.fnn = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x, enc_out):
        norm_x = self.attn_norm_1(x)
        # 掩码注意力
        x = x + self.mask_attn.forward(norm_x, norm_x, norm_x)
        # 多头注意力
        norm_x = self.attn_norm_2(x)
        h = x + self.attn.forward(norm_x, enc_out, enc_out)
        norm_h = self.fnn_norm(h)
        output = h + self.fnn.forward(norm_h)
        return output
    
class Decoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layers)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)
