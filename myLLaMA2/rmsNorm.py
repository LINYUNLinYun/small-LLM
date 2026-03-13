from torch import nn
import torch.nn.functional as F
import torch
import math
from modelConfig import ModelConfig

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: int = 1e-6):
        super().__init__()
        # 防止分母0
        self.eps = eps
        # 缩放参数
        self.gamma = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        # rsqrt求平方根的倒数 保持维度防止最后一个维度消失 (bsz seqlen dim) -> (bsz seqlen 1) 
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # 注意要先转成f32再传进去——考虑到模型一般不用f32训练，然后再转回来
        output = self._norm(x.float()).type_as(x)
        return self.gamma*output
    
if __name__ == '__main__':
    args = ModelConfig()
    norm = RMSNorm(args.dim, args.norm_eps)
    x = torch.randn(1, 50, args.dim)
    output = norm(x)
    print(output.shape)

    # out:
    # torch.Size([1, 50, 768])
