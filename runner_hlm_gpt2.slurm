#!/bin/bash

#SBATCH --job-name=hlm
#SBATCH --partition=RTX3090,RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:4
#SBATCH --mem=400G
#SBATCH --qos=high
#SBATCH --output=/home/yxwang/slurm/%j.out
#SBATCH --error=/home/yxwang/slurm/%j.err

bash train_hlm_gpt2.sh