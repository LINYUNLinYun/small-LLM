import torch.nn as nn
import torch.nn.functional as F
import torch
import math

from modelArgs import ModelArgs

class MultiHeadAttention(nn.Module):
    def __init__(self, args: ModelArgs, is_causal = False):
        super().__init__()
        # 断言检测 模型维度必须是头数量的整数倍
        assert args.dim % args.n_heads == 0
        self.dim = args.dim
        self.n_heads = args.n_heads
        self.head_dim = self.dim // self.n_heads
        self.is_causal = is_causal

        # wq wk wv权重矩阵，维度是 n_embd*dim，输入嵌入维度有时不等于dim
        # 其实原本应该是n_embd映射到head_dim，但是为了运用矩阵并行计算所以这里直接拼接为dim = n_heads*head_dim
        self.wq = nn.Linear(args.n_embd, self.dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.dim, bias=False)
        # 输出矩阵 维度为 dim x dim（head_dim = dim / n_heads）
        self.wo = nn.Linear(self.dim,self.dim,bias=False)
        # 注意力的dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的dropout
        self.res_dropout = nn.Dropout(args.dropout)

        # 因果注意力 mask
        if is_causal:
            # 生成一个 1*1*max_seq_len*max_seq_len第一个维度是方便BS广播相加，第二个是多头注意力
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            # 生成一个上三角矩阵, 主对角线不受影响
            mask = torch.triu(mask, diagonal=1)
            # 把mask挂载到对象 但不参与训练
            self.register_buffer("mask", mask)

    def forward(self, q: torch.tensor, k: torch.tensor, v: torch.tensor):
        # 获取批次大小和序列长度，[batch_size, seq_len, n_embd]
        bsz, seqlen, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, seqlen, n_embed) x (n_embed, dim)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, seqlen, n_head, head_dim)，然后交换维度，变成 (B, n_head, seqlen, head_dim)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么是先展开后转置
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 注意力分数
        # q*K^T/sigma -- (B, n_head, seqlen, seqlen)
        scores = torch.matmul(xq, xk.transpose(-2, -1))/math.sqrt(self.head_dim)

        # 掩码自注意力
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 要截断一下再把mask加上去
            scores = scores + self.mask[:,:,:seqlen,:seqlen]

        # 计算 softmax，维度为 (B, n_heads, seqlen, seqlen) 类型从f32到f16
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.attn_dropout(scores)
        # (B, n_heads, seqlen, seqlen) -> (B, n_head, seqlen, head_dim)
        output  = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, head_dim)，再拼接成 (B, T, dim)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1,2).contiguous().view((bsz,seqlen,self.dim))

        # 最终投影回原维度（外围会有个残差连接）。
        output = self.wo(output)
        output = self.res_dropout(output)
        return output



        
