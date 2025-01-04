#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=export
#SBATCH --partition=RTX3090,RTX4090 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=8  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=%j.out
#SBATCH --error=%j.err

echo doing $1

MODEL=neulab/gpt2-finetuned-wikitext103
# available splits: train/validation/test
python -u export_hs.py \
  --model_name_or_path ${MODEL} \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --output_dir checkpoints/${MODEL} \
  --do_eval --eval_subset $1 \
  --stride 1024 \
  --local_rank -1