#!/bin/bash

#SBATCH --job-name=1209vq2
#SBATCH --partition=RTX4090 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=10  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=/home/yxwang/slurm/%j.out
#SBATCH --error=/home/yxwang/slurm/%j.err

bash train.sh