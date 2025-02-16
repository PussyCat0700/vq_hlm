#!/bin/bash


export WANDB_DISABLED=true
export HF_ENDPOINT=https://hf-mirror.com
MODEL=neulab/gpt2-finetuned-wikitext103


# training type: 'codebook', 'full', 'after_input_layer', 'except_codebook'

vae_config_path=$1  # e.g. ./conf/models/residualvq_512.yaml
vae_model_path=$2  # e.g. ./runs/residualvq_ckpt1e3_codebookdim512_layer6_epoch3_mp/latest_checkpoint.pt
out_path=$3  # e.g. ./trained_models/test_pretrained_VAE

# CUDA_VISIBLE_DEVICES=0 
python run_train_hlm.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-103-raw-v1 \
    --model_name_or_path ${MODEL} \
    --model_type gpt2 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --vae_config_path $vae_config_path \
    --vae_pretrained_model_path $vae_model_path \
    --input_layers 6 \
    --ctx_layers 7 \
    --do_train \
    --weight_decay=0.1 \
    --save_strategy "epoch" \
    --warmup_steps=900 \
    --lr_scheduler_type="cosine" \
    --learning_rate 1e-3 \
    --logging_steps 10 \
    --fp16 \
    --output_dir $out_path \
    --overwrite_output_dir \
    --chunk_size 4 \
    --training_type after_input_layer
