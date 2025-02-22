import os
from transformers import AutoConfig, Trainer, TrainingArguments, pipeline
from datasets import Dataset, load_dataset
from models.gpt2 import HLMGPT2
import torch
import torch.nn.functional as F
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# 1. 加载数据集
def load_data(prefix):
    dataset = load_dataset(os.path.join(prefix, 'htoken.py'),trust_remote_code=True)
    return dataset

# 2. 配置GPT-2模型和Tokenizer
def load_model_and_tokenizer(model_name="gpt2"):
    gpt2_config = AutoConfig.from_pretrained(model_name)
    model = HLMGPT2(gpt2_config, model_name)
    return model

# 3. 数据预处理：tokenize数据集
def tokenize_data(dataset, tokenizer):
    # we dont need tokenization
    return dataset

# 4. 配置训练参数
def configure_training(model, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir="./exp/0222longepoch",          # 保存结果
        do_train=True,
        do_eval=True,
        eval_strategy='epoch',
        num_train_epochs=200,              # 训练轮数
        per_device_train_batch_size=2,   # 每个设备的训练批次大小
        gradient_accumulation_steps=16,   # 梯度累积步数
        per_device_eval_batch_size=2,    # 每个设备的评估批次大小
        logging_dir="./exp/0222longepoch",            # 日志目录
        logging_steps=10,               # 每500步记录日志
        save_steps=500,                  # 每500步保存模型
        learning_rate=1e-3,               # 学习率
        lr_scheduler_type="reduce_lr_on_plateau",        # 学习率调度器类型
        max_grad_norm=10,
        warmup_steps=1000,               # 预热步数
        weight_decay=0.01,              # 权重衰减
        adam_beta1=0.9,                      # Adam优化器的beta1参数
        adam_beta2=0.95,                 # Adam优化器的beta2参数
        ddp_find_unused_parameters=False,
    )
    
    trainer = Trainer(
        model=model,                        # 要训练的模型
        args=training_args,                 # 训练参数
        train_dataset=train_dataset,   # 训练数据集
        eval_dataset=val_dataset,       # 验证数据集
    )
    return trainer

# 5. 训练模型
def train_model(trainer):
    trainer.train()
    trainer.eval()

# 6. 保存模型
def save_model(model):
    model.save_pretrained("./gpt2_finetuned")

# 7. 生成文本
def generate_text(tokenizer):
    generator = pipeline("text-generation", model="./gpt2_finetuned", tokenizer=tokenizer)
    generated_text = generator("This is a test", max_length=50)
    print(generated_text)

def main():
    # 1. 加载数据
    logging.info('Loading data')
    data_path = '/home/yxwang/Dataset/vqhlm/ResidualSimVQ/'# 替换为你的数据文件路径
    dataset = load_data(data_path)
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset = dataset['test']
    
    # 2. 加载GPT-2模型和Tokenizer
    logging.info('Loading model and tokenizer')
    model_name = "/data1/public/hf/openai-community/gpt2"
    model = load_model_and_tokenizer(model_name)
    
    # 3. 数据预处理
    logging.info('Tokenizing data')
    # tokenized_datasets = tokenize_data(dataset, tokenizer)
    
    # 4. 配置训练参数
    logging.info('Configuring training')
    trainer = configure_training(model, train_dataset, val_dataset)
    
    # 5. 训练模型
    logging.info('Training model')
    train_model(trainer)
    
    # 6. 保存模型
    logging.info('Saving model and tokenizer')
    save_model(model)
    
    # 7. 生成文本
    logging.info('Generating text')
    # TODO
    # generate_text(tokenizer)

if __name__ == "__main__":
    main()
