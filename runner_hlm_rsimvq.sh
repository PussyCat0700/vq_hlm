#!/bin/bash

#SBATCH --account=bydai
#SBATCH --job-name=stage1
#SBATCH --partition=RTX4090 # 用sinfo命令可以看到所有队列
#SBATCH --ntasks-per-node=1 # 若多卡多进程，请调整此参数
#SBATCH --cpus-per-task=10  # 每个进程的CPU数量
#SBATCH --gres=gpu:1       # 若使用2块卡，则gres=gpu:2
 

export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export HF_ENDPOINT=https://hf-mirror.com
export WANDB_DISABLED=False

# wikitext-103
## wo_stride
# model_name=outputs/gpt2/wikitext/models/lm_gpt2_finetune_wk103_fp16_wostride/
## stride
# model_name=outputs/gpt2/wikitext/models/lm_gpt2_finetune_wk103_fp16_stride

# model_name=outputs/gpt2/redpajama/models/lm_gpt2_finetune_redpajama_bs16_fp16
# dataset=redpajama
# dataset_name="togethercomputer/RedPajama-Data-1T-Sample"
# task_name=context_${dataset}_gpt2_w4_l102_mse_d6_mse_cosine_ep10_lr1e-3
# wandb_project=Context-Decoder-Training
# dataset_config_name="None"

# model_name=outputs/gpt2/wikitext/models/lm_gpt2_wk103_from_scratch
# model_name=outputs/gpt2/wikitext/models/lm_gpt2_wk103_from_scratch_ep10
# dataset_name=wikitext
# dataset_config_name=wikitext-103-raw-v1
# wandb_project=Pretraining-Context-VQVAE-GPT-2-Wikitext103
# task_name=stage1_${dataset_name}_gpt2_ep10_fromscra_w4_l66_d6_mse_cosine_ep10_lr1e-3


# deepspeed --num_gpus 2 --master_port 29701 run_stage1.py \
#     --model_name_or_path ${model_name}\
#     --dataset_name ${dataset_name}\
#     --dataset_config_name ${dataset_config_name}\
#     --do_train --do_eval --eval_subset validation\
#     --num_train_epochs 10\
#     --deepspeed ./ds_zeros2_no_offload.json \
#     --per_device_train_batch_size 2 --per_device_eval_batch_size 1 --gradient_accumulation_steps 2\
#     --fp16 \
#     --learning_rate 1e-4\
#     --block_size 4096 --stride 1024\
#     --feature_extractor_layers 6 --decoder_layers 6 --n_layer 6 --w_size 4\
#     --report_to wandb --wandb_project ${wandb_project} --run_name ${task_name}  \
#     --output_dir outputs/gpt2/${dataset_name}/models/${task_name}/ \
#     --logging_steps 20 \
#     --save_strategy steps --save_steps 100 \
#     --eval_strategy steps --eval_steps 100 \
#     --save_total_limit 3\
#     >outputs/gpt2/${dataset_name}/logs/${task_name}.log

model_name_or_path='neulab/gpt2-finetuned-wikitext103'
dataset_name=wikitext
dataset_config_name=wikitext-103-raw-v1
wandb_project=Pretraining-Context-VQVAE-GPT-2-Wikitext103
task_name=stage1_w8_l66_d6_nll_mean_wohead_lr1e-3

python3 run_stage1.py \
    --model_name_or_path ${model_name_or_path}\
    --dataset_name ${dataset_name} --dataset_config_name ${dataset_config_name}  \
    --do_train --do_eval \
    --per_device_train_batch_size 2 \
    --learning_rate 1e-3\
    --save_total_limit 2\
    --block_size 8192\
    --feature_extractor_layers 6 --decoder_layers 6 --num_hidden_layers 6 --w_size 8\
    --overwrite_output_dir True\
    --report_to wandb --wandb_project ${wandb_project} --run_name ${task_name}  \
    --eval_strategy steps --eval_steps 100 \
    --output_dir outputs/gpt2/${dataset_name}/models/${task_name}/ \
    >outputs/gpt2/${dataset_name}/logs/${task_name}.log