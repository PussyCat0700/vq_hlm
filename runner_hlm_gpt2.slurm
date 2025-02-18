#!/bin/bash

#SBATCH --job-name=hlm
#SBATCH --partition=A100,ADA6000,L40S
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --qos=high
#SBATCH --output=/home/yxwang/slurm/%j.out
#SBATCH --error=/home/yxwang/slurm/%j.err
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=luckywang@sjtu.edu.cn

export NCCL_P2P_DISABLE=1
bash train_hlm_gpt2.sh