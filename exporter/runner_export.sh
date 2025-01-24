MODEL=/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/models/gpt2_wiki
# available splits: train/validation/test
python -u export_hs.py \
  --model_name_or_path ${MODEL} \
  --dataset_name /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/liuyuliang-240108350135/datasets/wikitext103 \
  --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --do_eval \
  --local_rank -1