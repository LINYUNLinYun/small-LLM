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

# 测试jsonl读取函数
if __name__ == "__main__":
    dataset_path = "./seq-monkey/mobvoi_seq_monkey_general_open_corpus.jsonl"
    count = 0
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
    