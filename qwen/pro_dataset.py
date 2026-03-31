# 读取.env文件
import os
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from itertools import chain

model_path = './models/Qwen2.5-1.5B'

tokenizer = AutoTokenizer.from_pretrained(model_path)


load_dotenv()

dataset_path = os.getenv("dataset_path")
pretrain_file = os.getenv("pretrain_file_path")
test_pretrain_file = os.getenv("test_pretrain_file_path")

def tokenizer_function(examples):
    # 返回分词的文本
    return tokenizer([i for i in examples['text']])

# 预训练一般将文本拼接成固定长度的文本段，因为一次性学习多个样本序列语义不影响性能
# 这里原来为2048，试图解决oom问题改为
block_size = 512
def group_texts(examples):
    # 将文本段拼接起来 *在内存级别把二维列表拆开，chain合并成一个
    concatenated_examples = {k : list(chain(*examples[k])) for k in examples.keys()}
    # 计算长度
    total_lenght = len(concatenated_examples[list(examples.keys())[0]])
    # print(total_lenght)     # bs_size 个seqlen的和
    if total_lenght >= block_size:
        total_lenght = (total_lenght // block_size) * block_size    # 化为整数倍

    result = {
        k : [ t[i: i+block_size] for i in range(0, total_lenght, block_size)] for k, t in concatenated_examples.items()
    }
    result['labels'] = result['input_ids'].copy()       # CLM中，输入即标签， 毕竟是在预测自己
    return result

def pro_dataset():
    # 加载测试数据集 -- 测试
    dataset = load_dataset('json', data_files = os.path.join(dataset_path, test_pretrain_file))
    # 打印dataset的列
    print(dataset.column_names)
    # print(dataset["train"][0])
    # print([i for i in dataset["train"][0]['text']])
    # 查看特征
    column_names = list(dataset["train"].features)
    # print(column_names)
    # columnes_name:["text"]
    tokenized_dataset = dataset.map(         # dim of input_ids : (samples_num, seq_len)
        function= tokenizer_function,           # attention mask 是全1的
        batched=True,
        num_proc= 10,
        remove_columns= column_names,
        load_from_cache_file= True,
        desc= "Running tokenizer on dataset",
    )
    # group_text(tokenizeried_dataset['train'][:10])

    # 批量处理
    lm_datasets = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=10,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
        batch_size = 40000,
    )
    train_dataset = lm_datasets["train"]
    return train_dataset

def main():
    # 加载测试数据集 -- 测试
    dataset = load_dataset('json', data_files = os.path.join(dataset_path, test_pretrain_file))
    # 打印dataset的列
    print(dataset.column_names)
    # print(dataset["train"][0])
    # print([i for i in dataset["train"][0]['text']])
    # 查看特征
    column_names = list(dataset["train"].features)
    # print(column_names)
    # columnes_name:["text"]
    tokenized_dataset = dataset.map(         # dim of input_ids : (samples_num, seq_len)
        function= tokenizer_function,           # attention mask 是全1的
        batched=True,
        num_proc= 10,
        remove_columns= column_names,
        load_from_cache_file= True,
        desc= "Running tokenizer on dataset",
    )
    # group_text(tokenizeried_dataset['train'][:10])

    # 批量处理
    lm_datasets = tokenized_dataset.map(
        group_texts,
        batched=True,
        num_proc=10,
        load_from_cache_file=True,
        desc=f"Grouping texts in chunks of {block_size}",
        batch_size = 40000,
    )
    train_dataset = lm_datasets["train"]
    # print(tokenizeried_dataset.column_names)
    # print(tokenizeried_dataset['train']['attention_mask'])


if __name__ == "__main__":
    main()