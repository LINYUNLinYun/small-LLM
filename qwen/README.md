# 大模型训练实践

## 下载模型
环境变量设置在进程内存中，子进程可共享。
```shell
cd qwen && python download_model.py
```

## 预训练
```shell
python pretrain.py
```

很bad的是，即使是512的blocksize，1的batchsize, bf16的，在4090(24G)依然oom，难道1.5B的模型只能做到这一步吗？

是的，很遗憾，1.5B的模型光参数就要约3GB，即使是512的blocksize，一个bs也要2G，优化器又占用12GB。所以全量微调是不现实的，除非用8bit优化器？？（有这种东西吗）

### deepspeed分布式训练与 ZeRO优化器
“ZeRO” 代表 Zero Redundancy Optimizer（零冗余优化器）

## sft微调
修改了若干错误。。

## LORA微调

### LoRA 原理
emmmm暂时省略。只需记住，在wq和wv上应用效果比较好。


$$
Output=Wx+ \frac{\alpha}{r}(BAx)
$$

- $\alpha$: LoRA 的缩放因子，主要用来lora的作用程度。越大，lora的作用贡献程度越大；而且能在不训练lora参数的情况下，直接通过调整 $\alpha$ 来控制 lora 的影响。
- $r$: LoRA 的秩，表示低秩矩阵的维度。这个缩放因子和r的比值通常控制在1-2

### 代码实现
1. 确定要应用lora的层，peft 库目前支持调用 LoRA 的层包括：nn.Linear、nn.Embedding、nn.Conv2d 三种。
2. 在这些层的基础上增加一个旁路，模拟参数更新
3. 冻结原参数

打印一下模型结构：
```python
model = AutoModelForCausalLM.from_pretrained(model_path)

print(model)
```

结果如下，可以看到wq和wv都是同输入输出维度不变的线性层，适合应用lora。而kv权重矩阵由于头数比较少，因此有个6倍的维度压缩。
```
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(151936, 1536)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2SdpaAttention(
          (q_proj): Linear(in_features=1536, out_features=1536, bias=True)
          (k_proj): Linear(in_features=1536, out_features=256, bias=True)
          (v_proj): Linear(in_features=1536, out_features=256, bias=True)
          (o_proj): Linear(in_features=1536, out_features=1536, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (up_proj): Linear(in_features=1536, out_features=8960, bias=False)
          (down_proj): Linear(in_features=8960, out_features=1536, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm()
        (post_attention_layernorm): Qwen2RMSNorm()
      )
    )
    (norm): Qwen2RMSNorm()
  )
  (lm_head): Linear(in_features=1536, out_features=151936, bias=False)
)
```

那么这个q_proj和v_proj就是我们应用lora的目标。  

```python
# 定义一个lora层，主要是定义了一些参数
class LoraLayer:
    def __init__(
        self,
        r: int, # LoRA 的秩
        lora_alpha: int, # 归一化参数
        lora_dropout: float, # LoRA 层的 dropout 比例
        merge_weights: bool, # eval 模式中，是否将 LoRA 矩阵的值加到原权重矩阵上
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False

```
- 如果启用了merge_weights，那么在eval模式下会将lora矩阵的值加到原权重矩阵上，避免了LoRA权重和原权重相加的开销，推理速度会快一点。
- 因为lora微调可能会过拟合一些小数据集，所以dropout参数需要设置，默认是恒等映射。

有个问题是，在self.reset_parameters()中，lora_a和lora_b会被按照规定的初始化方式初始化吗？即A矩阵使用kaiming_uniform，B矩阵使用0初始化。这个初始化会不会将原来的权重也覆盖了？

```python
    def reset_parameters(self):
        # Initialization as per the paper
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)
```


还有个有趣的地方在前馈函数：
```python
    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            # 如果已经合并
            if self.r > 0 and self.merged:
                # 这里实际上是dim, r * r, dim的矩阵乘法，因为pytorch的线性层矩阵是(out_features, in_features)存储的 
                self.weight.data -= (
                    transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
                self.merged = False
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        # 不禁用lora
        elif self.r > 0 and not self.merged:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return result
        else:
            return F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
```

这里为什么要做这个转置呢？因为pytorch的线性层矩阵是(out_features, in_features)存储的——这个是由于历史原因导致，如计算机的行优先存储方便读取，数学形式，以及gpu的反向传播优化等多种因素。

所以对与pytorch的线性层，实际上做的是运算(batch_size, in_features) @ (in_features, out_features) = (batch_size, out_features)
$$
Output = xW^T + b
$$