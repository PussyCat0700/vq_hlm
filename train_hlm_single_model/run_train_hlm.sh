MODEL=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/gpt2_wiki
export WANDB_DISABLED=true

# training type: 'codebook', 'full', 'after_input_layer', 'except_codebook'

# CUDA_VISIBLE_DEVICES=0 
python run_train_hlm.py \
    --dataset_name /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/datasets/wikitext103 \
    --dataset_config_name wikitext-103-raw-v1 \
    --model_name_or_path ${MODEL} \
    --model_type gpt2 \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --vae_config_path ../conf/models/residualvq_512.yaml \
    --vae_pretrained_model_path ../runs/residualvq_ckpt1e3_codebookdim512_layer6_epoch3_mp/latest_checkpoint.pt \
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
    --output_dir ./trained_models/test_pretrained_VAE \
    --overwrite_output_dir \
    --chunk_size 4 \
    --training_type after_input_layer
