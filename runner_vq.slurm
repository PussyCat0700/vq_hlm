#!/bin/bash

#SBATCH --job-name=vqhlm
#SBATCH --partition=RTX3090,RTX4090
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --qos=high
#SBATCH --depend=singleton
#SBATCH --output=/home/yxwang/slurm/%j.out
#SBATCH --error=/home/yxwang/slurm/%j.err

bash train.sh $1