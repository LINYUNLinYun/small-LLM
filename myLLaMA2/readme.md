# 动手实现LLaMA2
## 定义超参数
[modelConfig.py](./modelConfig.py)继承了transformers库中的参数类，方便后续导出huggingface参数模型

## 实现RMSNorm
公式：
$$
\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2 + \epsilon}} \cdot \gamma
$$

[rmsNorm.py](./rmsNorm.py)实现了RMSNorm类，层归一化

## repeat kv
在transformer架构中，qkv的维度是一样的。不存在维度缩放的问题。在LLaMa2中，为了拯救 GPU 显存（解决 KV Cache 爆炸问题），引入了 repeat_kv 参数。该参数控制了在计算注意力时，键（K）和值（V）是否重复使用。具体来说，先看transformer的mha：
```python
        # 为什么是先展开后转置
        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_heads, self.head_dim)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
```
这里补充下：由于数据在显存中是连续存储的，view并没有改变数据物理位置，只是逻辑上把数据看作"token A 的第一个头、token A 的第二个头、...、token B 的第一个头、token B 的第二个头、..."。如果直接view成(bsz, self.n_heads, seqlen, self.head_dim)，就会把数据看作"token A 的第一个头、token B 的第一个头、...、token A 的第二个头、token B 的第二个头、..."，这样在计算注意力时就会出问题。所以一定要先view再转置

根据transformer的mha实现，可以发现它的qkv拥有一样的头，这会导致一个问题：当输入序列长度很长时，kv cache的开销会很恐怖。因此有了MQA(Multi Query Attention)，它的kv只有一个头而Q是多头的。但是这样性能就会下降。而GQA(Grouped Query Attention)就是LLaMA的折中方法，LLaMA 会把 Q 的头进行**分组**。比如Q有32个头，而K和V的头有8个，每四个Q头一组共用一个K和V头。降低了 KV Cache 的开销。

[repeat_kv.py](./repeat_kv.py)就是负责把kv临时复制出份数来，让它和Q对齐。

## 旋转位置编码
旋转位置编码（Rotary Position Embedding，简称 RoPE）。
旋转嵌入（RoPE）是目前大语言模型（如 LLaMA、GLM 等）中最主流、最核心的位置编码方式。

和transformer一样具有外推性。
$$
f(\mathbf{q}, m) = \text{Rot}(\mathbf{q}, m) = \begin{pmatrix}
\cos m\theta_0 & -\sin m\theta_0 & 0 & 0 & \dots \
\sin m\theta_0 & \cos m\theta_0 & 0 & 0 & \dots \
0 & 0 & \cos m\theta_1 & -\sin m\theta_1 & \dots \
0 & 0 & \sin m\theta_1 & \cos m\theta_1 & \dots \
\vdots & \vdots & \vdots & \vdots & \ddots
\end{pmatrix}
\begin{pmatrix} q_0 \ q_1 \ q_2 \ q_3 \ \vdots \end{pmatrix}
$$