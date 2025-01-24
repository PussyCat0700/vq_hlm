export WANDB_DISABLED=true

#!/bin/bash

# 定义模型列表（可以替换为你的模型路径）
MODEL_LIST=(
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/gpt2_wiki"
  "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/vqvae_epo10_lr5e4_cb1024_mask_no_freezing_layer36/checkpoint-893"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/vqvae_epo10_lr5e4_cb1024_mask_no_freezing_layer36/checkpoint-1786"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/vqvae_epo10_lr5e4_cb1024_mask_no_freezing_layer36/checkpoint-2679"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/vqvae_epo10_lr5e4_cb1024_mask_no_freezing_layer36/checkpoint-3572"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/vqvae_epo10_lr5e4_cb1024_mask_no_freezing_layer36/checkpoint-4465"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/vqvae_epo10_lr5e4_cb1024_mask_no_freezing_layer36/checkpoint-5358"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/vqvae_epo10_lr5e4_mask/checkpoint-6249"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/vqvae_epo10_lr5e4_mask/checkpoint-7142"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/vqvae_epo10_lr5e4_mask/checkpoint-8034"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/vqvae_epo10_lr5e4_mask/checkpoint-8920"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/test_save_vqvae_epo10/checkpoint-6000"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/test_save_vqvae_epo10/checkpoint-7000"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/test_save_vqvae_epo10/checkpoint-8000"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/test_save_vqvae_epo10/checkpoint-9000"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/test_save_vqvae_epo10/checkpoint-4000"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/test_save_vqvae_epo10/checkpoint-5000"

  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/full_gpt2_epo10_lr1e3_bz64_10_epo_co_training_code_V10/"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/full_gpt2_epo10_lr1e3_bz128_frozen_first_6_layers_10_epo_group8_co_training_code_V10/"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/full_gpt2_epo10_lr1e3_bz128_frozen_first_6_layers_10_epo_group16_co_training_code_V10/"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/full_gpt2_epo10_lr1e3_bz128_frozen_first_6_layers_10_epo_group2_co_training_code_V10/"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/full_gpt2_epo10_lr5e4_bz128_frozen_first_6_layers_10_epo_co_training_code_V10/"
  # "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/ctx_token_construction/knn-transformers-master/trained_models/full_gpt2_epo10_lr1e3_bz128_frozen_first_6_layers_10_epo_co_training_code_V10/"
  )

# 数据集路径
DATASET_NAME="/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/datasets/wikitext103"
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
    --input_layers 3 \
    --ctx_layers 6

  echo "Evaluation for ${MODEL} completed."
done
