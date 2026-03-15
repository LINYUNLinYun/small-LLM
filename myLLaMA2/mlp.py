from torch import nn
import torch.nn.functional as F
import torch
import math
from modelConfig import ModelConfig



class MLP(nn.Module):
    def __init__(self, dim:int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        # 控制参数量并向上取整
        if hidden_dim is None:
            hidden_dim = dim*4
            hidden_dim = int(hidden_dim*2/3)
            hidden_dim = multiple_of*((hidden_dim + multiple_of -1)//multiple_of)

        # 线性映射
        self.w1 = nn.Linear(dim, hidden_dim,bias= False)
        # 映射回去
        self.w2 = nn.Linear(hidden_dim, dim,bias= False)
        # 门控线性单元
        self.w3 = nn.Linear(dim, hidden_dim,bias= False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # 这里x经过w1和silu激活，和x经过w3的结果相乘，最后经过w2映射回模型维度，并且进行dropout
        return self.dropout(self.w2(F.silu(self.w1(x))*self.w3(x)))

if __name__ == '__main__':
    # 参数
    args = ModelConfig()
    # 创建MLP实例
    mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
    # 随机生成数据
    x = torch.randn(1, 50, args.dim)
    # 运行MLP模型
    output = mlp(x)
    print(output.shape)

    # out:
    # torch.Size([1, 50, 768])

        