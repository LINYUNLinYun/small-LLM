from torch import nn
import torch.nn.functional as F
import torch
import math
from typing import Tuple
from modelConfig import ModelConfig

def repeat_kv(x: torch.Tensor, n_rep: torch.Tensor) -> torch.Tensor:
    # 这里因为kv的头数和q不一样，要做区分
    bsz, seqlen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    # 先在第四维前新增一个维度，再用expand扩展成 (bsz, seqlen, n_kv_heads, n_rep, head_dim)
    x = x[:, :, :, None, :].expand(bsz, seqlen, n_kv_heads, n_rep, head_dim)
    return x.reshape((bsz, seqlen, n_kv_heads*n_rep, head_dim)) #最后QKV具有一样的维度

# 注意：此处的dim应为 dim//n_head，因为我们是对每个head进行旋转嵌入
def precompute_freqs(dim: int, max_seqlen: int, theta: float = 10000.0):
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = torch.exp(
            math.log(theta) * (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
        )
    # freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 生成一个从0到max_seqlen的序列，步长为1，长度为max_seqlen，就是pos
    t = torch.arange(max_seqlen, device=freqs.device)
    # 计算外积，得到一个二维矩阵，其实就是(max_seqlen,1)*(1,dim//2)->(max_seqlen, dim//2)，每个元素是位置和频率的乘积
    freqs = torch.outer(t, freqs).float()
    # 计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos, freqs_sin

def reshape_rot_angle(freqs_cis: torch.Tensor, x: torch.Tensor):
    # 获取x的维度：4维？ 答案-- (bsz, seqlen, n_heads, head_dim//2)
    ndim = x.ndim
    
    # 断言，确保1在x的维度范围内
    assert 0 <= 1 < ndim
    # 断言，确保freqs_cis的形状与x的第二维和最后一维相同
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    
    # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，即(max_seqlen, head_dim//2)变为(1, max_seqlen, 1, head_dim//2)
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    # 将freqs_cis调整为新的形状，并返回
    return freqs_cis.view(shape)

def RoPE(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # 将查询和键张量转换为浮点数，然后拆成d/2对
    # 先获取除了最后一维以外的维度-- (bsz, seqlen, n_heads)，然后在最后一维上新增维度，变成(bsz, seqlen, n_heads, head_dim//2, 2)，最后用unbind将最后一个维度拆成实部和虚部
    # unbind(-1)会将最后一个维度解除绑定，返回两个张量：一个是x轴一个是y轴
    # 所以这里每个轴的维度是(bsz, seqlen, n_heads, head_dim//2)，因为我们是对每个head进行旋转嵌入的
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播 (1, seqlen, 1, head_dim//2)
    freqs_cos = reshape_rot_angle(freqs_cos, xq_r)
    freqs_sin = reshape_rot_angle(freqs_sin, xq_r)

    # 应用旋转，二位旋转矩阵的方程形式
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


if __name__ == '__main__':
    args = ModelConfig()
    # randn函数，生成一个形状为(1, 50, n_kv_heads, dim//n_heads)的张量，模拟注意力机制中的键值对输入
    x = torch.randn(1, 50, args.n_kv_heads, args.dim//args.n_heads)
    n_rep = args.n_heads // args.n_kv_heads
    output = repeat_kv(x, n_rep)
    print(output.shape)

    # out:
    # torch.Size([1, 50, 16, 48])

    # 验证Rope
    freqs_cos, freqs_sin = precompute_freqs(args.dim//args.n_heads, args.max_seq_len)
    print(freqs_cos.shape, freqs_sin.shape)
    xq = torch.randn(1, args.max_seq_len, args.n_heads, args.dim//args.n_heads)
    xk = torch.randn(1, args.max_seq_len, args.n_kv_heads, args.dim//args.n_heads)
    xq_out, xk_out = RoPE(xq, xk, freqs_cos, freqs_sin)
    print(xq_out.shape, xk_out.shape)

    # out:
    # torch.Size(1, 512, 16, 48) torch.Size(1, 512, 8, 48)s