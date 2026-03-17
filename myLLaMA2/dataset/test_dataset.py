import json
from typing import Generator


def read_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """读取JSONL文件并安全提取文本数据"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'text' not in data:
                    raise KeyError(f"Missing 'text' field in line {line_num}")
                yield data['text']
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as e:
                print(e)
                continue


def read_conversations_from_jsonl(file_path: str) -> Generator[list[dict], None, None]:
    """读取SFT数据并提取conversations字段"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line)
                if 'conversations' not in data:
                    raise KeyError(f"Missing 'conversations' field in line {line_num}")
                yield data['conversations']
            except json.JSONDecodeError:
                print(f"Error decoding JSON in line {line_num}")
                continue
            except KeyError as e:
                print(e)
                continue

# 截取jsonl文件的前几个并保存为新的jsonl文件，方便测试
def save_sample_jsonl(input_path: str, output_path: str, num_samples: int = 5) -> None:
    """从输入JSONL文件中提取前num_samples行并保存到新的JSONL文件"""
    with open(output_path, 'w', encoding='utf-8') as out_file:
        with open(input_path, 'r', encoding='utf-8') as in_file:
            for line_num, line in enumerate(in_file, 1):
                if line_num > num_samples:
                    break
                out_file.write(line)

# 测试jsonl读取函数
if __name__ == "__main__":
    dataset_path = "input/seq_monkey_dealed.jsonl"
    count = 0

    # sample_output_path = "input/sample_pretrain.jsonl"
    # save_sample_jsonl(dataset_path, sample_output_path, num_samples=100)
    # exit()
    for text in read_texts_from_jsonl(dataset_path):
        print(f"Pretrain sample {count + 1}:")
        print(text)
        print()
        count += 1
        if count >= 5:  # 只打印前5行文本进行测试
            break
    # 测试sft数据集,json格式
    sft_dataset_path = "seq-monkey/BelleGroup/train_3.5M_CN.json"
    count = 0
    for conversations in read_conversations_from_jsonl(sft_dataset_path):
        print(f"SFT sample {count + 1}:")
        for turn in conversations:
            role = turn.get('from', 'unknown')
            value = turn.get('value', '')
            print(f"{role}: {value}")
        print()
        count += 1
        if count >= 5:  # 只打印前5条记录进行测试
            break
    