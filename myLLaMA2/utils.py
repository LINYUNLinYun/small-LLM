from torch import nn
import torch.nn.functional as F
import torch
import math
from modelConfig import ModelConfig

def repeat_kv(x: torch.Tensor, n_rep: torch.Tensor) -> torch.Tensor:
    # 这里因为kv的头数和q不一样，要做区分
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 先在第四维前新增一个维度，再用expand扩展成 (bsz, seqlen, n_kv_heads, n_rep, head_dim)
    return x.reshape((bsz, seqlen, n_kv_heads*n_rep, head_dim)) #最后QKV具有一样的维度

# 注意：此处的dim应为 dim//n_head，因为我们是对每个head进行旋转嵌入
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = -math.log(theta) * torch.arange(0, dim, 2)[: (dim // 2)].float() / dim
    # freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从0到end的序列，长度为end
    t = torch.arange(end, device=freqs.device)
    # 计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin


if __name__ == '__main__':
    args = ModelConfig()
    # randn函数，生成一个形状为(1, 50, n_kv_heads, dim//n_heads)的张量，模拟注意力机制中的键值对输入
    x = torch.randn(1, 50, args.n_kv_heads, args.dim//args.n_heads)
    n_rep = args.n_heads // args.n_kv_heads
    output = repeat_kv(x, n_rep)
    print(output.shape)

    # out:
    # torch.Size([1, 50, 16, 48])