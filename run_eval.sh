#!/bin/bash


export WANDB_DISABLED=true
export HF_ENDPOINT=https://hf-mirror.com


MODEL_LIST=(
  # Replace with your model here
  # e.g.
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/gpt2_wiki"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/test_pretrained_VAE/checkpoint-3"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/vq_hlm/train_hlm_single_model/trained_models/test_pretrained_VAE/checkpoint-6"
  )

# 数据集路径
DATASET_NAME="wikitext"
DATASET_CONFIG_NAME="wikitext-103-raw-v1"
OUTPUT_DIR_BASE="checkpoints"

# 循环遍历模型列表
for MODEL in "${MODEL_LIST[@]}"; do
  # 输出当前模型路径
  echo "Evaluating model: ${MODEL}"

  # 运行评估脚本
  CUDA_VISIBLE_DEVICES=0 python -u run_clm_eval.py \
    --model_name_or_path ${MODEL} \
    --dataset_name ${DATASET_NAME} \
    --dataset_config_name ${DATASET_CONFIG_NAME} \
    --output_dir "${OUTPUT_DIR_BASE}/${MODEL//\//_}" \
    --do_eval \
    --eval_subset test \
    --input_layers 6 \
    --ctx_layers 9 \
    --vae_config_path ./conf/models/residualvq_512.yaml \
    --vae_pretrained_model_path ./runs/residualvq_ckpt1e3_codebookdim512_layer6_epoch3_mp/latest_checkpoint.pt \
    --stride 1024

  echo "Evaluation for ${MODEL} completed."
done
