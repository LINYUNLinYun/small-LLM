from dataclasses import dataclass

@dataclass
class ModelArgs:
    dim: int           # 模型的核心隐藏层维度 (Model dimension)
    n_layers: int      # Transformer 的层数
    n_heads: int       # 注意力头的数量
    n_embd: int        # 输入嵌入的维度 (有时等于 dim)
    max_seq_len: int   # 最大序列长度 (用于位置编码或掩码)
    dropout: float = 0.1 # Dropout 比率
    vocab_size: int = None # 词表大小 (如果需要的话)
    block_size: int = None # 块大小 (如果需要的话)

# 使用示例：
# args = ModelArgs(
#     dim=512, 
#     n_layers=6, 
#     n_heads=8, 
#     n_embd=512, 
#     max_seq_len=2048,
#     dropout=0.1
# )