## Transformer
### 注意力
注意力机制的成名架构（其实注意力机制最早在CV领域被提出）

对于一个Query，有一个词向量$q$（也许也能称之为查询向量），维度是$1\times d$。这个d是词向量空间的维度，具体看词向量空间为多少。然后key键值对应的词向量为$k$，如果把词向量空间的所有key堆叠起来（方便运算）就得到了$K$矩阵，维度为$n \times d$，这个n取决于词向量空间具有多少个词（n）。

所以对于一个Query，计算他和每一个键的相似程度，其实就是通过点积计算词之间的相似程度。
$$
x = qK^T
$$

得到的这个x就反映了Query和所有key的相似度程度，再做个softmax的归一化，所有权重的加和为1，也就是**注意力分数**。再把注意力分数和值向量做乘积即可。   

考虑到一次性可以做多个Query操作，因此可以把q堆叠到一起形成$Q$矩阵，维度为$m \times d$。得到以下的最终公式：

$$
attention(Q,K,V)=softmax(QK^T)V
$$
中间那个做矩阵乘法后，维度变为$m \times n$。
再考虑到，随着词向量的维度增加，我们设为$d_k$，点积的结果会变得很大（因为点积结果的方差实际上是和$d_k$线性增长的）会导致softmax的梯度失效。假设向量中的元素均值为0，方差为1。所以还要再做个放缩，把点积的Var化为1。
$$
attention(Q,K,V)=softmax( \frac{QK^T}{\sqrt{d_k}})V 
$$
即，
$$
Var( \frac{QK^T}{\sqrt{d_k}}) = \frac{1}{d_k}Var( QK^T) = \frac{d_k}{d_k} = 1
$$

注意力机制的pytorch实现：
```python

```

### 自注意力机制
注意力机制是在一个目标序列和源序列之间查相似度（两个完全不同的序列）。自注意力是一个序列内部寻找相似度。Q、K、V全部来自同一个序列。它们都是由同一个输入数据（通过乘以不同的权重矩阵）变化而来的。
> Q、K、V 都由同一个输入通过不同的参数矩阵计算得到。在 Encoder 中，Q、K、V 分别是输入对参数矩阵$ W_q、W_k、W_v$做积得到，从而拟合输入语句中每一个 token 对其他所有 token 的关系。(这隐含着什么呢？QKV竟然都是从一个东西那里输出来的？？？)
- 注意力机制 解决对齐或映射问题，建立两个不同**语言/模态**之间的桥梁。
- 自注意力机制 解决的内部依赖或特征提取问题。发掘的是句子内部的语法语义关系。



### 掩码自注意力
确保了并行训练的时候，信息的单向流动，模型不会提前获取到未来的信息。即所谓的因果性？

==这里有个有意思的是，训练是并行的，但是推理好像只能是串行的。因为训练其实已经提前知道标准答案了。==

```
<BOS> 【MASK】【MASK】【MASK】【MASK】
<BOS>    I   【MASK】 【MASK】【MASK】
<BOS>    I     like  【MASK】【MASK】
<BOS>    I     like    you  【MASK】
<BOS>    I     like    you   </EOS>
```

所以要生成一个上三角矩阵，用来遮蔽标准答案。考虑到模型的输入维度为（batch_size, seq_len, hidden_size），batchsize是一步输入的数据组（有多少的句），seqlen是一个序列的长度（有多少的词），hiddensize是词向量的维度（一个词被映射成一个向量）？？

总之， Mask 矩阵维度一般为 (1, seq_len, seq_len)。

> 主对角线以下都是零的方阵称为上三角矩阵(右上方非零，左下方为0)

mask的torch实现，有个很诡异的事情是，scores矩阵的维度应该是$m \times n$但是在这里scores的大小变为（seqlen, seqlen），这大概是因为自注意力机制只关注句子内部的关系，所以成了个和句子长度大小一样边长的方阵：
```python
# 创建一个上三角矩阵，用于遮蔽未来信息。
# 先通过 full 函数创建一个 1 * seq_len * seq_len 的矩阵
mask = torch.full((1, args.max_seq_len, args.max_seq_len), float("-inf"))
# triu 函数的功能是创建一个上三角矩阵，diag参数会对主对角线产生影响，可以print看看
mask = torch.triu(mask, diagonal=1)
# 此处的 scores 为计算得到的注意力分数，mask 为上文生成的掩码矩阵
scores = scores + mask[:, :seqlen, :seqlen]
scores = F.softmax(scores.float(), dim=-1).type_as(xq)

```

### 多头注意力
“横看成岭侧成峰，远近高低各不同”。在transformer中，一般有8个头，每个头关注的信息不同（拟合不同的关系）如语法结构、上下文关联等，将其综合可得到最全面的拟合信息。

公式：
$$MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$$
$$where \quad head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$$

对于headi都有分别的Wi矩阵确保即使是一样的QKV也能关注到不同的东西。随后把不同头的自注意力处理拼接起来，在经过一层线性层处理得到最终的输出。
> 通过三个组合矩阵来代替了n个参数矩阵的组合，其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积
多头注意力机制的代码实现：
```python
import torch.nn as nn
import torch

'''多头自注意力计算模块'''
class MultiHeadAttention(nn.Module):

    def __init__(self, args: ModelArgs, is_causal=False):
        # 构造函数
        # args: 配置对象
        super().__init__()
        # 隐藏层维度必须是头数的整数倍，因为后面我们会将输入拆成头数个矩阵
        assert args.dim % args.n_heads == 0
        # 每个头的维度，等于模型维度除以头的总数。下取整除法很诡异，上面都有断言为啥还要取整除法？
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads

        # Wq, Wk, Wv 参数矩阵，每个参数矩阵为 n_embd x dim
        # 这里通过三个组合矩阵来代替了n个参数矩阵的组合（列拼接），其逻辑在于矩阵内积再拼接其实等同于拼接矩阵再内积，
        # 不理解的读者可以自行模拟一下，每一个线性层其实相当于n个参数矩阵的拼接
        self.wq = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.n_embd, self.n_heads * self.head_dim, bias=False)
        # 输出权重矩阵，维度为 dim x dim（head_dim = dim / n_heads）
        self.wo = nn.Linear(self.n_heads * self.head_dim, args.dim, bias=False)
        # 注意力的 dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        # 残差连接的 dropout
        self.resid_dropout = nn.Dropout(args.dropout)
        self.is_causal = is_causal

        # 创建一个上三角矩阵，用于遮蔽未来信息
        # 注意，因为是多头注意力，Mask 矩阵比之前我们定义的多一个维度
        if is_causal:
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            # 注册为模型的缓冲区
            self.register_buffer("mask", mask)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):

        # 获取批次大小和序列长度，[batch_size, seq_len, dim]
        bsz, seqlen, _ = q.shape

        # 计算查询（Q）、键（K）、值（V）,输入通过参数矩阵层，维度为 (B, T, n_embed) x (n_embed, dim) -> (B, T, dim)
        xq, xk, xv = self.wq(q), self.wk(k), self.wv(v)

        # 将 Q、K、V 拆分成多头，维度为 (B, T, n_head, dim // n_head)，然后交换维度，变成 (B, n_head, T, dim // n_head)
        # 因为在注意力计算中我们是取了后两个维度参与计算
        # 为什么要先按B*T*n_head*C//n_head展开再互换1、2维度而不是直接按注意力输入展开，是因为view的展开方式是直接把输入全部排开，
        # 然后按要求构造，可以发现只有上述操作能够实现我们将每个头对应部分取出来的目标
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # 注意力计算
        # 计算 QK^T / sqrt(d_k)，维度为 (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 掩码自注意力必须有注意力掩码
        if self.is_causal:
            assert hasattr(self, 'mask')
            # 这里截取到序列长度，因为有些序列可能比 max_seq_len 短
            scores = scores + self.mask[:, :, :seqlen, :seqlen]
        # 计算 softmax，维度为 (B, nh, T, T)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        # 做 Dropout
        scores = self.attn_dropout(scores)
        # V * Score，维度为(B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        output = torch.matmul(scores, xv)

        # 恢复时间维度并合并头。
        # 将多头的结果拼接起来, 先交换维度为 (B, T, n_head, dim // n_head)，再拼接成 (B, T, n_head * dim // n_head)
        # contiguous 函数用于重新开辟一块新内存存储，因为Pytorch设置先transpose再view会报错，
        # 因为view直接基于底层存储得到，然而transpose并不会改变底层存储，因此需要额外存储
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # 最终投影回残差流。
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output

```


## Encoder-Decoder
Transformer架构由编码器和解码器组成。编码，就是将输入的自然语言序列通过隐藏层编码成能够表征语义的向量（或矩阵），解码自然就是反过来。
后续是encoder和decode内部的一些传统神经网络架构。
### 前馈神经网络
简称FNN，实现比较简单。每一个encoder layer包含一个注意力机制和一个FNN。以下是代码实现：
```python

class MLP(nn.Module):
    '''前馈神经网络——多层感知机'''
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        # 定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        # 定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 前向传播函数
        # 首先，输入x通过第一层线性变换和RELU激活函数
        # 最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.relu(self.w1(x))))
    
```
Transformer 的前馈神经网络由**两个线性层中间加一个 RELU 激活函数**组成的，以及前馈神经网络还加入了一个 Dropout 层来防止过拟合。**Dropout 层只在训练时开启**，推理/测试阶段关闭，所以许多Transformer结构示意图中不会画出该层。

### 层归一化
- 批归一化：常用于图像处理。是把不同样本的某个维度求均值然后归一化。
- 层归一化：常用于自然语言处理。是把一个样本的所有维度求均值然后归一化。

批归一化，对第i个样本的第j个维度求均值：
$$
\mu_j = \frac{1}{m}\sum_{i=1}^{m}Z_j^i
$$
$$
\sigma_j^2 = \frac{1}{m}\sum_{i=1}^{m}(Z_j^i-\mu_j)^2
$$
化为**标准**正态分布：
$$
\hat{Z_j} = \frac{Z_j - \mu_j}{\sqrt{\sigma^2 + \epsilon}}
$$
> 但是，批归一化存在一些缺陷，例如：当显存有限，mini-batch 较小时，Batch Norm 取的样本的均值和方差不能反映全局的统计分布信息，从而导致效果变差；对于在时间维度展开的 RNN，不同句子的同一分布大概率不同，所以 Batch Norm 的归一化会失去意义；在训练时，Batch Norm 需要保存每个 step 的统计信息（均值和方差）。在测试时，由于变长句子的特性，测试集可能出现比训练集更长的句子，所以对于后面位置的 step，是没有训练的统计量使用的；应用 Batch Norm，每个 step 都需要去保存和计算 batch 统计量，耗时又耗力

因此有了更好的layer norm的实现，统计一个样本的不同层的均值和方差：
```python
class LayerNorm(nn.Module):
    ''' Layer Norm 层'''
    def __init__(self, features, eps=1e-6):
        super().__init__()
        # 线性矩阵做映射
        self.a_2 = nn.Parameter(torch.ones(features))   # 初始权重向量1
        self.b_2 = nn.Parameter(torch.zeros(features))  # 0
        self.eps = eps
    
    def forward(self, x):
        # 在统计每个样本所有维度的值，求均值和方差
        # 对最后一个维度做均值 [bsz, max_len, embed_dim] -> [bsz, max_len, 1]
        mean = x.mean(-1, keepdim=True) # mean: [bsz, max_len, 1] 保持维度因为后面还要广播
        std = x.std(-1, keepdim=True) # std: [bsz, max_len, 1]
        # 注意这里也在最后一个维度发生了广播
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2  # 给缩放是因为纯标准分布学习的效果不好

```

### 残差连接
这个简单，
$$
x=x+MultiHeadSelfAttention(LayerNorm(x))
$$
$$
output=x+FNN(LayerNorm(x))
$$

```python
# 注意力计算
h = x + self.attention.forward(self.attention_norm(x))
# 经过前馈神经网络
out = h + self.feed_forward.forward(self.fnn_norm(h))
```

### Encoder
我们先搭建一个encoder layer，每个layer包含一个attention和一个FNN，且两者前都要跟norm
```python
class EncoderLayer(nn.Module):
  '''Encoder层'''
    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有两个 LayerNorm，分别在 Attention 之前和 MLP 之前
        self.attention_norm = LayerNorm(args.n_embd)
        # Encoder 不需要掩码，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.fnn_norm = LayerNorm(args.n_embd)
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x):
        # Layer Norm
        norm_x = self.attention_norm(x)
        # 自注意力
        h = x + self.attention.forward(norm_x, norm_x, norm_x)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.fnn_norm(h))
        return out

```
然后可以构造Encoder，一个encoder由N个layer构成:
```python
class Encoder(nn.Module):
    '''Encoder 块'''
    def __init__(self, args):
        # 构造函数可用新写法 super().__init__()
        super(Encoder, self).__init__() 
        # 一个 Encoder 由 N 个 Encoder Layer 组成
        self.layers = nn.ModuleList([EncoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x):
        #分别通过 N 层 Encoder Layer
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


```

### Decoder 
类似的，我们也可以先搭建 Decoder Layer，再将 N 个 Decoder Layer 组装为 Decoder。但是和 Encoder 不同的是，Decoder 由两个注意力层和一个前馈神经网络组成。第一个注意力层是一个**掩码自注意力层**，即使用 Mask 的注意力计算，保证每一个 token 只能使用该 token 之前的注意力分数；第二个注意力层是一个**多头注意力层**，该层将使用第一个注意力层的输出作为 query，使用 Encoder 的输出作为 key 和 value，来计算注意力分数。最后，再经过前馈神经网络：
```python
class DecoderLayer(nn.Module):
  '''解码层'''
    def __init__(self, args):
        super().__init__()
        # 一个 Layer 中有三个 LayerNorm，分别在 Mask Attention 之前、Self Attention 之前和 MLP 之前
        self.attention_norm_1 = LayerNorm(args.n_embd)
        # Decoder 的第一个部分是 Mask Attention，传入 is_causal=True
        self.mask_attention = MultiHeadAttention(args, is_causal=True)
        self.attention_norm_2 = LayerNorm(args.n_embd)
        # Decoder 的第二个部分是 类似于 Encoder 的 Attention，传入 is_causal=False
        self.attention = MultiHeadAttention(args, is_causal=False)
        self.ffn_norm = LayerNorm(args.n_embd)
        # 第三个部分是 MLP
        self.feed_forward = MLP(args.dim, args.dim, args.dropout)

    def forward(self, x, enc_out):
        # Layer Norm
        norm_x = self.attention_norm_1(x)
        # 掩码自注意力
        x = x + self.mask_attention.forward(norm_x, norm_x, norm_x)
        # 多头注意力
        norm_x = self.attention_norm_2(x)
        h = x + self.attention.forward(norm_x, enc_out, enc_out)
        # 经过前馈神经网络
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
```
Decoder块：
```python
class Decoder(nn.Module):
    '''解码器'''
    def __init__(self, args):
        super(Decoder, self).__init__() #同上
        # 一个 Decoder 由 N 个 Decoder Layer 组成
        self.layers = nn.ModuleList([DecoderLayer(args) for _ in range(args.n_layer)])
        self.norm = LayerNorm(args.n_embd)

    def forward(self, x, enc_out):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, enc_out)
        return self.norm(x)

```

## 搭建一个Transformer的其他组件

### embedding 层
Embedding 层的输入往往是一个形状为 （batch_size，seq_len，1）的矩阵，第一个维度是一次批处理的数量，第二个维度是自然语言序列的长度，第三个维度则是 token 经过 tokenizer 转化成的 index 值。而 Embedding 内部其实是一个可训练的（Vocab_size，embedding_dim）的权重矩阵，**词表里的每一个值，都对应一行维度为 embedding_dim 的向量**。对于输入的值，会对应到这个词向量，然后**拼接成（batch_size，seq_len，embedding_dim）的矩阵输出**。直接调用`self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)`即可。

### 位置编码

注意力机制实现很好的并行计算的同时，要解决位置编码的问题。如lstm这种递归处理的自然没有位置编码的需求，因为输入的顺序即为序列的顺序。

位置编码，即根据序列中 token 的相对位置对其进行编码，再将位置编码加入词向量编码中。位置编码的方式有很多，Transformer 使用了正余弦函数来进行位置编码，其编码方式为：
$$
PE(pos,2i)=sin(pos/10000^{2i/d_{model}})\\
PE(pos,2i+1)=cos(pos/10000^{2i/d_{model}})  \\
$$
其中，i是向量内部的index，取决于嵌入空间的维度（也就是d of model），取值范围是$[0, d_model/2+1]$；Pos是词在句子中的位置。
最大的特别就是对于奇数和偶数位置的词向量采用了不同的三角函数。分母还有个特殊的地方在于，在i小的地方，分母接近于0；分母大的地方，分母接近于1。

$$
\begin{gathered}
\mathrm{x}=
\begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 \\
0.2 & 0.3 & 0.4 & 0.5 \\
0.3 & 0.4 & 0.5 & 0.6 \\
0.4 & 0.5 & 0.6 & 0.7
\end{bmatrix} \\
% \text{则经过位置编码后的词向量为:} \\
\mathrm{x}_{\mathrm{PE}}=
\begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 \\
0.2 & 0.3 & 0.4 & 0.5 \\
0.3 & 0.4 & 0.5 & 0.6 \\
0.4 & 0.5 & 0.6 & 0.7
\end{bmatrix}+
\begin{bmatrix}
\sin(\frac{0}{10000^0}) & \cos(\frac{0}{10000^0}) & \sin(\frac{0}{10000^{1/4}}) & \cos(\frac{0}{10000^{1/4}}) \\
\sin(\frac{1}{10000^0}) & \cos(\frac{1}{10000^0}) & \sin(\frac{1}{10000^{1/4}}) & \cos(\frac{1}{10000^{1/4}}) \\
\sin(\frac{2}{10000^0}) & \cos(\frac{2}{10000^0}) & \sin(\frac{2}{10000^{1/4}}) & \cos(\frac{2}{10000^{1/4}}) \\
\sin(\frac{3}{10000^0}) & \cos(\frac{3}{10000^0}) & \sin(\frac{3}{10000^{1/4}}) & \cos(\frac{3}{10000^{1/4}})
\end{bmatrix}=
\begin{bmatrix}
0.1 & 1.2 & 0.3 & 1.4 \\
1.041 & 0.84 & 0.41 & 1.49 \\
1.209 & -0.016 & 0.52 & 1.59 \\
0.541 & -0.489 & 0.895 & 1.655
\end{bmatrix}
\end{gathered}
$$
详细的数学证明见[happyllm第二章](https://datawhalechina.github.io/happy-llm/#/./chapter2/%E7%AC%AC%E4%BA%8C%E7%AB%A0%20Transformer%E6%9E%B6%E6%9E%84?id=_225-encoder)；

它的好处在于：
- 外推性 PE 能够适应比训练集里面所有句子更长的句子
- 可以让模型容易地计算出相对位置，对于固定长度的间距 k可利用三角和公式求解
- 位置编码虽然由周期函数构成 但不存在重复的可能。其一为编码函数传入的是整数步长，永远无法接近2$\pi$的的倍数；其二为对于一个n维的嵌入空间，每个向量有n/2个不同波长的“向量时钟”，他们组合出来的是无限的状态空间，因此绝对找不出两组重复的编码。

numpy实现：
```python
def PositionEncoding(seq_len, d_model, n=10000):
    P = np.zeros((seq_len, d_model)) # 创建一个全 0 的空白矩阵
    for k in range(seq_len):         # 外层循环：遍历每一个词的位置（也就是公式里的 pos）
        for i in np.arange(int(d_model/2)): # 内层循环：遍历维度的一半（因为一次填两个坑）
            
            # 严格按照公式计算分母： 10000^(2i / d_model)
            denominator = np.power(n, 2*i/d_model) 
            
            # 给偶数列 (2i) 填入 sin 值
            P[k, 2*i] = np.sin(k/denominator)
            # 给奇数列 (2i+1) 填入 cos 值
            P[k, 2*i+1] = np.cos(k/denominator)
    return Ps
```

pytorch实现：
```python
class PositionalEncoding(nn.Module):
    '''位置编码模块'''
    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        # self.dropout = nn.Dropout(p=args.dropout)

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

```