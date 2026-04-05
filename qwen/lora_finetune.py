import logging
import math
import os
import sys
from dataclasses import dataclass, field
from torch.utils.data.datapipes.iter import IterableWrapper     # api changed
from itertools import chain
from typing import Optional,List
import re
import torch
import torch.nn as nn
from torch.nn import functional as F
# from torch import transpose

import datasets
import pandas as pd
import torch
from datasets import load_dataset
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
import datetime
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
import swanlab
from tqdm import tqdm
# from finetune import ModelArguments, DataTrainingArguments

import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from transformers import Trainer
import json
from torch.utils.data import Dataset
from swanlab.integration.huggingface import SwanLabCallback
# from finetune import SupervisedDataset

# 超参类
@dataclass
class ModelArguments:
    """
    关于模型的参数
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "预训练模型参数地址"
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "模型训练使用的数据类型，推荐 bfloat16"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )


@dataclass
class DataTrainingArguments:
    """
    关于训练的参数
    """

    train_files: Optional[str]  = field(default=None, metadata={"help": "训练数据路径"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "最大文本块长度"
            )
        },
    )

# 指令文本处理
# 参考：https://github.com/QwenLM/Qwen/blob/main/finetune.py
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
from typing import Dict

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

logger = logging.getLogger(__name__)

target_modules = ["q_proj", "v_proj"]

# 找到模型的各个组件中，名字里带"q_proj"，"v_proj"的
# target_module_found = re.fullmatch(self.peft_config.target_modules, key)
# 这里的 key，是模型的组件名

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
        # 禁用lora
        self.disable_adapters = False

# 继承了两个父类 融合了下nn的线性层和lora层
class Linear(nn.Linear, LoraLayer):
    # LoRA 层
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False, 
        merge_weights: bool = True,
        **kwargs,
    ):
        # 调用父类的构造函数
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.fan_in_fan_out = fan_in_fan_out
        # Actual trainable parameters
        if self.r > 0:
            # 参数矩阵 A
            self.lora_A = nn.Linear(in_features, self.r, bias=False)
            # 参数矩阵 B
            self.lora_B = nn.Linear(self.r, out_features, bias=False)
            # 归一化系数
            self.scaling = self.lora_alpha / self.r
            # 冻结原参数，仅更新 A 和 B
            self.weight.requires_grad = False
        # 初始化参数（AB）
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T
    @staticmethod
    def transpose(weight: torch.Tensor, fan_in_fan_out):
        # 如果是 fan_in_fan_out 模式，则转置权重矩阵
        return weight.transpose(0, 1) if fan_in_fan_out else weight
    
    # 前馈函数
    def forward(self, x: torch.Tensor):
        if self.disable_adapters:
            # 如果已经合并
            if self.r > 0 and self.merged:
                # 这里实际上是dim, r * r, dim的矩阵乘法，因为pytorch的线性层矩阵是(out_features, in_features)存储的 
                self.weight.data -= (
                    self.transpose(self.lora_B.weight @ self.lora_A.weight, self.fan_in_fan_out) * self.scaling
                )
                self.merged = False
            return F.linear(x, self.transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
        # 不禁用lora
        elif self.r > 0 and not self.merged:
            result = F.linear(x, self.transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            if self.r > 0:
                result += self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
            return result
        else:
            return F.linear(x, self.transpose(self.weight, self.fan_in_fan_out), bias=self.bias)



if __name__ == '__main__':
    # model_path = './models/Qwen2.5-1.5B'

    # 加载脚本参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 初始化 SwanLab
    # swanlab.init(project="lora_sft", experiment_name="qwen2.5-1.5B")

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

    # 加载基座模型
    model = AutoModelForCausalLM.from_pretrained(
        # model_args.model_name_or_path,trust_remote_code=True,attn_implementation="flash_attention_2",)
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=getattr(torch, model_args.torch_dtype) if model_args.torch_dtype else None)
    
    model.enable_input_require_grads()
    
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"继承一个预训练模型 - Total size={n_params/2**20:.2f}M params")

    # 初始化 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    logger.info("完成 tokenzier 加载")

    # model = AutoModelForCausalLM.from_pretrained(model_path)

    # 加载基座模型
    # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

    peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
        )
    
    model = get_peft_model(model, peft_config)

    # 加载微调数据
    with open(data_args.train_files) as f:
        lst = [json.loads(line) for line in f.readlines()[:10000]]      # 只取前10000条？？？
    logger.info("完成训练集加载")
    logger.info(f"训练集地址：{data_args.train_files}")
    logger.info(f'训练样本总数:{len(lst)}')
    # logger.info(f"训练集采样：{ds["train"][0]}")

    train_dataset = SupervisedDataset(lst, tokenizer=tokenizer, max_len=2048)
    
    logger.info("初始化 Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= IterableWrapper(train_dataset),
        tokenizer=tokenizer,
        callbacks=[swanlab_callback], # 添加 SwanLab 回调
    )
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset= IterableWrapper(train_dataset),
    #     tokenizer=tokenizer
    # )
    trainer.train()


    # print(model)
