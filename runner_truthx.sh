#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=truthx
#SBATCH --partition=RTX3090,RTX4090,A100 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=./runs/truthx/%j.out
#SBATCH --error=./runs/truthx/%j.err

set -e

ckpt_dir=./runs/truthx
echo $ckpt_dir

python train_vq.py --ckpt_dir $ckpt_dir --model_config ./conf/models/truthx_residualvq.yaml
python train_vq.py --ckpt_dir $ckpt_dir --model_config ./conf/models/truthx_residualvq.yaml --test