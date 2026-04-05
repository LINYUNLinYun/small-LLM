"""
Qwen2.5-1.5B 轻量化 LoRA 微调脚本
=====================================
技术栈说明：
- QLoRA (Quantized LoRA): 4-bit 量化 + LoRA 微调，大幅降低显存占用
- 梯度检查点 (Gradient Checkpointing): 用时间换空间，减少激活值显存
- 混合精度训练 (BF16): 利用 4090 的 Tensor Core 加速训练
- 梯度累积 (Gradient Accumulation): 模拟更大的 batch size

适用硬件：RTX 4090 (24GB 显存)
预计显存占用：约 8-12GB（取决于 batch_size 和 max_seq_length）
"""

# ============================================================
# 第一部分：导入必要的库
# ============================================================

import os                          # 操作系统接口，用于读取环境变量、路径操作等
import json                        # JSON 处理，用于加载数据集
import logging                     # 日志模块，用于记录训练过程信息
import sys                         # 系统模块，用于标准输出等
from dataclasses import dataclass, field  # 数据类装饰器，用于定义配置类
from typing import Optional, Dict        # 类型提示，表示参数可以是某种类型或 None

import torch                       # PyTorch 深度学习框架
from torch.utils.data import Dataset       # PyTorch 数据集基类
from torch.utils.data.datapipes.iter import IterableWrapper  # 可迭代数据集包装器

from datasets import load_dataset          # HuggingFace 数据集加载工具
from itertools import chain                # 迭代器工具，用于拼接列表
from tqdm import tqdm                      # 进度条显示库

# Transformers 库：HuggingFace 提供的预训练模型工具包
from transformers import (
    AutoConfig,                    # 自动加载模型配置
    AutoModelForCausalLM,          # 自动加载因果语言模型（用于文本生成）
    AutoTokenizer,                 # 自动加载分词器
    HfArgumentParser,              # 参数解析器，用于命令行参数
    Trainer,                       # 训练器，封装了训练循环
    TrainingArguments,             # 训练参数配置类
    default_data_collator,         # 默认数据整理器，用于将样本打包成 batch
    set_seed,                      # 设置随机种子，保证实验可复现
    BitsAndBytesConfig,            # 量化配置类，用于 4-bit/8-bit 量化
)

# PEFT 库：Parameter-Efficient Fine-Tuning，参数高效微调库
from peft import (
    LoraConfig,                    # LoRA 配置类
    get_peft_model,                # 将 LoRA 应用到模型上的函数
    PeftModel,                     # PEFT 模型基类
    TaskType,                      # 任务类型枚举
    prepare_model_for_kbit_training,  # 为量化模型做准备的函数
)

from transformers.trainer_utils import get_last_checkpoint  # 用于查找最新的 checkpoint
from dotenv import load_dotenv     # 加载 .env 文件中的环境变量

import swanlab                     # SwanLab：实验跟踪工具，类似 TensorBoard
from swanlab.integration.huggingface import SwanLabCallback

# ============================================================
# 第二部分：设置日志配置
# ============================================================

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# ============================================================
# 第三部分：定义参数配置类
# ============================================================

@dataclass
class ModelArguments:
    """
    模型相关参数配置类
    使用 @dataclass 装饰器自动生成 __init__ 方法
    这些参数可以通过命令行传入，也可以使用默认值
    """
    
    model_name_or_path: Optional[str] = field(
        default="./models/Qwen2.5-1.5B",  # 默认模型路径
        metadata={
            "help": "预训练模型的路径，可以是本地路径或 HuggingFace 模型名称"
        },
    )
    
    torch_dtype: Optional[str] = field(
        default="bfloat16",  # 使用 BF16 精度，4090 支持
        metadata={
            "help": "模型训练使用的数据类型，推荐 bfloat16（4090 支持）",
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )

@dataclass
class DataTrainingArguments:
    """
    数据训练相关参数配置类
    """
    
    train_files: Optional[str] = field(
        default=None,
        metadata={"help": "训练数据的 JSON 文件路径"},
    )
    
    max_seq_length: Optional[int] = field(
        default=512,  # 序列长度设为 512，平衡显存和效果
        metadata={
            "help": "模型输入的最大序列长度，越大显存占用越多"
        },
    )
    
    dataset_sample_ratio: Optional[float] = field(
        default=1.0,  # 默认使用 100% 数据
        metadata={
            "help": "数据集采样比例，范围 0.0-1.0。例如 0.1 表示只使用 10% 的数据进行训练，可以大幅缩短训练时间"
        },
    )
    
    num_proc: Optional[int] = field(
        default=4,  # 数据预处理使用的进程数
        metadata={
            "help": "数据预处理时的并行进程数，根据你的 CPU 核心数调整"
        },
    )

@dataclass
class LoraArguments:
    """
    LoRA 相关参数配置类
    LoRA 的核心思想：不更新原始权重，而是在旁边添加一个低秩矩阵来模拟参数更新
    公式：W' = W + BA，其中 W 是冻结的原始权重，B 和 A 是可训练的低秩矩阵
    """
    
    lora_r: int = field(
        default=16,  # LoRA 的秩，越大表达能力越强，但参数量也越多
        metadata={
            "help": "LoRA 的秩（rank），表示低秩矩阵的维度。常见值：8, 16, 32。越大效果越好但显存占用越多"
        },
    )
    
    lora_alpha: int = field(
        default=32,  # LoRA 的缩放系数，通常是 lora_r 的 2 倍
        metadata={
            "help": "LoRA 的缩放系数（alpha），控制 LoRA 更新的强度。通常设置为 lora_r 的 2 倍"
        },
    )
    
    lora_dropout: float = field(
        default=0.05,  # Dropout 率，防止过拟合
        metadata={
            "help": "LoRA 层的 dropout 率，防止过拟合。常见值：0.05, 0.1"
        },
    )
    
    target_modules: Optional[str] = field(
        default="all-linear",  # 对所有线性层应用 LoRA
        metadata={
            "help": "要应用 LoRA 的目标模块名称。'all-linear' 表示所有线性层，也可以指定具体层名如 'q_proj,v_proj'"
        },
    )

# ============================================================
# 第四部分：数据预处理函数
# ============================================================

def preprocess(sources, tokenizer, max_len, system_message: str = "You are a helpful assistant."):
    # prompt 模板
    roles = {"human": "<|im_start|>human", "assistant": "<|im_start|>assistant"}

    # 不同的 tokenizer 需要特别定义
    # BOS
    im_start = tokenizer("<|im_start|>").input_ids
    # EOS
    im_end = tokenizer("<|im_end|>").input_ids
    # PAD
    IGNORE_TOKEN_ID = tokenizer.pad_token_id
    # 换行符
    nl_tokens = tokenizer('\n').input_ids
    # 角色标识符
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('human').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # 拼接多轮对话
    input_ids, targets = [], []
    for i in tqdm(range(len(sources))):
        source = sources[i]
        # 从 user 开始
        if source[0]["from"] != "human":
            source = source[1:]
        # 分别是输入和输出
        input_id, target = [], []
        # system: 【BOS】system\nYou are a helpful assistant.【EOS】\n
        system = im_start + _system + tokenizer(system_message).input_ids + im_end + nl_tokens
        input_id += system
        # system 不需要拟合
        target += im_start + [IGNORE_TOKEN_ID] * (len(system)-3) + im_end + nl_tokens
        assert len(input_id) == len(target)
        # 依次拼接
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            # user：<|im_start|>human\ninstruction【EOS】\n
            # assistant：<|im_start|>assistant\nresponse【EOS】\n
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                tokenizer(sentence["value"]).input_ids + im_end + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>human':
                # user 不需要拟合
                _target = im_start + [IGNORE_TOKEN_ID] * (len(_input_id)-3) + im_end + nl_tokens
            elif role == '<|im_start|>assistant':
                # assistant 需要拟合
                _target = im_start + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                    _input_id[len(tokenizer(role).input_ids)+1:-2] + im_end + nl_tokens
            else:
                print(role)
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        # 最后进行 PAD
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    # print(input_ids)
    input_ids = torch.tensor(input_ids)
    targets = torch.tensor(targets)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )
# 自定义一个 Dataset
class SupervisedDataset(Dataset):

    def __init__(self, raw_data, tokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()
        # 加载并预处理数据
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


if __name__ == '__main__':
    # 加载脚本参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, LoraArguments, TrainingArguments))
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()

    # 初始化 SwanLab
    swanlab_callback = SwanLabCallback(
        project="lora_sft",
        experiment_name="qwen2.5-1.5B",
        config=training_args, # 把训练参数也记录进去
    )
    
    # 设置日志
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # 将日志级别设置为 INFO
    transformers.utils.logging.set_verbosity_info()
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # 训练整体情况记录
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # 检查 checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"输出路径 ({training_args.output_dir}) 非空 "
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"从 {last_checkpoint}恢复训练"
            )

    # 设置随机数种子.
    set_seed(training_args.seed)

    # 初始化模型
    logger.warning("加载预训练模型")
    logger.info(f"模型参数地址：{model_args.model_name_or_path}")

    # 量化配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else None,
        quantization_config=bnb_config if training_args.fp16 or training_args.bf16 else None, # 使用量化
    )
    
    # 为 kbit 训练准备模型
    model = prepare_model_for_kbit_training(model)
    model.enable_input_require_grads() # 允许梯度计算
    
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"继承一个预训练模型 - Total size={n_params/2**20:.2f}M params")

    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    logger.info("完成 tokenzier 加载")
    
    # 设置 tokenizer 的 pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA 配置
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=[m.strip() for m in lora_args.target_modules.split(',')] if lora_args.target_modules != "all-linear" else lora_args.target_modules,
        inference_mode=False,
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # 加载微调数据
    with open(data_args.train_files, "r", encoding="utf-8") as f:
        raw_datasets = [json.loads(line) for line in f.readlines()]
    logger.info("完成训练集加载")
    logger.info(f"训练集地址：{data_args.train_files}")
    logger.info(f'训练样本总数:{len(raw_datasets)}')
    
    # 采样数据集
    if data_args.dataset_sample_ratio < 1.0:
        sample_size = int(len(raw_datasets) * data_args.dataset_sample_ratio)
        raw_datasets = raw_datasets[:sample_size]
        logger.info(f"数据集采样，使用 {data_args.dataset_sample_ratio*100:.2f}% 的数据，样本数：{len(raw_datasets)}")

    train_dataset = SupervisedDataset(raw_datasets, tokenizer=tokenizer, max_len=data_args.max_seq_length)
    
    logger.info("初始化 Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        callbacks=[swanlab_callback], # 添加 SwanLab 回调
    )

    # 从 checkpoint 加载
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
            checkpoint = last_checkpoint

    logger.info("开始训练")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model() 

    # 保存训练指标
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info("训练完成")