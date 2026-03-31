from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = './models/Qwen2.5-1.5B'

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

from transformers import TrainingArguments
from transformers import Trainer, default_data_collator

from torch.utils.data import IterableDataset
from pro_dataset import pro_dataset

train_dataset = pro_dataset()

def test():

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir="output",# 训练参数输出路径
        per_device_train_batch_size=1,# 训练的 batch_size, 改为1试图解决oom
        gradient_accumulation_steps=16,# 梯度累计步数，实际 bs = 设置的 bs * 累计步数, 16/*1=16
        logging_steps=1,# 打印 loss 的步数间隔 测试置为1
        num_train_epochs=1,# 训练的 epoch 数
        bf16 = True,  # 使用 bf16 训练, 试图解决oom问题
        save_steps=100, # 保存模型参数的步数间隔
        learning_rate=1e-4,# 学习率
        gradient_checkpointing=True# 开启梯度检查点
    )


    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset= train_dataset,
        eval_dataset= None,
        tokenizer=tokenizer,
        # 默认为 MLM 的 collator，使用 CLM 的 collater
        data_collator=default_data_collator
    )
    # 实际的 batch size = 4(batch_size) * 4(累计步数) = 16
    # 如果你的总数据量少于 16 条，Trainer 一步都走不完就不会打印 Loss！

    print("数据集的数据类型是:", type(train_dataset))
    if isinstance(train_dataset, list):
        print("数据集总长度是:", len(train_dataset))
    trainer.train()
if __name__ == "__main__":
    # 开始训练 ， 只进行一个短暂的训练，测试流程是否正确
    test()

