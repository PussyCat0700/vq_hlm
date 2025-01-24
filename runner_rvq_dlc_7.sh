#!/bin/bash

#SBATCH --account=yfliu3
#SBATCH --job-name=residualvq
#SBATCH --partition=RTX3090,RTX4090,A100 # 用sinfo命令可以看到所有队列
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2
#SBATCH --output=./runs/residualvq/%j.out
#SBATCH --error=./runs/residualvq/%j.err

# CUDA_VISIBLE_DEVICES=1 
python train_vq.py \
 --data_config ./conf/data/norm_layer6.yaml \
 --previous_model_path ./runs/residualvq_ckpt1e3_codebookdim640_layer6_epo3/best_checkpoint.pt \
 --ckpt_dir ./runs/residualvq_ckpt1e3_codebookdim640_layer6_epo3_ckpt_lr1e4_epo100 \
 --model_config conf/models/residualvq_640.yaml \
 --lr 1e-4 \
 --train_epochs 100 \
#  --data_config \
#  --model_config \

# python train_vq.py --ckpt_dir ./runs/residualvq --model_config conf/models/residualvq.yaml --test