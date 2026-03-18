import os
import json
from tqdm import tqdm

# pretrain_data 为运行download_dataset.sh时，下载的pretrain_data本地路径
pretrain_data = 'seq-monkey/mobvoi_seq_monkey_general_open_corpus.jsonl'
output_pretrain_data = 'input/seq_monkey_dealed_2.jsonl'

# sft_data 为运行download_dataset.sh时，下载的sft_data本地路径
sft_data = 'seq-monkey/BelleGroup/train_3.5M_CN.json'
output_sft_data = 'input/BelleGroup_sft.jsonl'

# 1 处理预训练数据
def split_text(text, chunk_size=512):
    """将文本按指定长度切分成块"""
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def process_pretrain_data(input_path, output_path):
    with open(output_path, 'a', encoding='utf-8') as pretrain:
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc=f"Processing lines in {input_path}", unit="lines", leave=False):
                line = json.loads(line)
                text = line['text']
                chunks = split_text(text)
                for chunk in chunks:
                    pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')

# 2 处理SFT数据
def convert_message(data):
    """
    将原始数据转换为标准格式
    """
    message = [
        {"role": "system", "content": "你是一个AI助手"},
    ]
    for item in data:
        if item['from'] == 'human':
            message.append({'role': 'user', 'content': item['value']})
        elif item['from'] == 'assistant':
            message.append({'role': 'assistant', 'content': item['value']})
    return message

def process_sft_data(input_path, output_path):
    with open(output_path, 'a', encoding='utf-8') as sft:
        with open(input_path, 'r', encoding='utf-8') as f:
            for item in tqdm(f, desc="Processing", unit="lines"):
                item = json.loads(item)
                message = convert_message(item['conversations'])
                sft.write(json.dumps(message, ensure_ascii=False) + '\n')


def main():
    process_pretrain_data(pretrain_data, output_pretrain_data)
    # process_sft_data(sft_data, output_sft_data)


if __name__ == '__main__':
    main()