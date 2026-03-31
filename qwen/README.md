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

## 微调