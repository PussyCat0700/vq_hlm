#!/bin/bash

#SBATCH --job-name=hlm
#SBATCH --partition=RTX3090,RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --qos=high
#SBATCH --output=/home/yxwang/slurm/%j.out
#SBATCH --error=/home/yxwang/slurm/%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=luckywang@sjtu.edu.cn

bash train_hlm_gpt2.sh