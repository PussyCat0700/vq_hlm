import os
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
from datasets import Dataset, load_dataset
import torch
import torch.nn.functional as F
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

# 0108_rvq_CN24_CD1k_CS8k_LR1E-3_EPOCH100_BS256
CODEBOOK_DIM=1024
EMBEDDING_DIM=768
CODEBOOK_SIZE=8192
NUM_QUANTIZER=24

# 1. 加载数据集
def load_data(prefix):
    dataset = load_dataset(os.path.join(prefix, 'htoken.py'),trust_remote_code=True)
    return dataset

# 2. 配置GPT-2模型和Tokenizer
def load_model_and_tokenizer(model_name="gpt2"):
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.wte = [torch.nn.Embedding(CODEBOOK_SIZE, EMBEDDING_DIM) for _ in range(NUM_QUANTIZER)]
    # model.wte_proj = torch.nn.Linear(CODEBOOK_DIM, EMBEDDING_DIM)
    model.vqhead = [torch.nn.Linear(EMBEDDING_DIM, CODEBOOK_SIZE) for _ in range(NUM_QUANTIZER)]

    # TODO, use vqvae as lm_head
    # model.vqvae = 

    model.transformer.wte = torch.nn.Identity()
    model.lm_head = torch.nn.Identity()
    return model

# 3. 数据预处理：tokenize数据集
def tokenize_data(dataset, tokenizer):
    # def tokenize_function(examples):
    #     return tokenizer(examples['text'], return_tensors='pt', padding=True, truncation=True)
    
    # tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # return tokenized_datasets
    return dataset

# 4. 配置训练参数
class train_hlm(Trainer):
    def compute_loss(self, model, inputs, num_items_in_batch):
        for i in range(NUM_QUANTIZER):
            model.wte[i].to(model.device)
            model.vqhead[i].to(model.device)
        # inputs: input_ids: Bx1024x24, label: Bx1024x24
        input_ids, label = inputs['input_ids'], inputs['labels']
        input_embeds = None
        for i in range(NUM_QUANTIZER):
            if input_embeds is None:
                input_embeds = model.wte[i](input_ids[:, :, i])
            else:
                input_embeds += model.wte[i](input_ids[:, :, i])
        # input_embeds = model.wte_proj(input_embeds)
        outputs = model.transformer(
            inputs_embeds=input_embeds,
        )
        hidden_states = outputs[0]
        lm_logits = []
        for i in range(NUM_QUANTIZER):
            lm_logits.append(model.vqhead[i](hidden_states)) # Bx1024x8192
        lm_logits = torch.stack(lm_logits, dim=2)
        loss = F.cross_entropy(lm_logits.view(-1, lm_logits.size(-1)), label.view(-1).long(), ignore_index=-100)
        return loss

def configure_training(model, train_dataset, val_dataset):
    training_args = TrainingArguments(
        output_dir="./exp/0214testhlm",          # 保存结果
        do_train=True,
        do_eval=True,
        num_train_epochs=10,              # 训练轮数
        per_device_train_batch_size=4,   # 每个设备的训练批次大小
        gradient_accumulation_steps=8,   # 梯度累积步数
        per_device_eval_batch_size=4,    # 每个设备的评估批次大小
        logging_dir="./exp/0214testhlm",            # 日志目录
        logging_steps=10,               # 每500步记录日志
        save_steps=100,                  # 每500步保存模型
        learning_rate=1e-4,               # 学习率
        warmup_steps=100,               # 预热步数
        weight_decay=0.01,              # 权重衰减
        adam_beta1=0.9,                      # Adam优化器的beta1参数
        adam_beta2=0.95,                 # Adam优化器的beta2参数
    )
    
    trainer = train_hlm(
        model=model,                        # 要训练的模型
        args=training_args,                 # 训练参数
        train_dataset=train_dataset,   # 训练数据集
        eval_dataset=val_dataset,       # 验证数据集
    )
    return trainer

# 5. 训练模型
def train_model(trainer):
    trainer.train()

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
    data_path = '/home/yxwang/Dataset/vqhlm/wikitext103_gpt2ln2_stride1024/'# 替换为你的数据文件路径
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
    # generate_text(tokenizer)

if __name__ == "__main__":
    main()
