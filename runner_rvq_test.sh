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

# CUDA_VISIBLE_DEVICES=1 python train_vq.py \
#  --data_config ./conf/data/no_norm_layer6.yaml \
#  --ckpt_dir ./runs/residualvq_nonorm_ckpt5e3_layer10_epo10 \
#  --model_config conf/models/residualvq.yaml \
#  --lr 5e-3 \
#  --train_epochs 1 \
# #  --data_config \
# #  --model_config \

# CUDA_VISIBLE_DEVICES=1 
# train_vq_meanpooling
# train_vq_last
python train_vq_meanpooling.py \
 --ckpt_dir ./runs/residualvq_ckpt1e3_codebookdim512_layer6_epoch3_view_lr_mean \
 --model_config conf/models/residualvq_512.yaml \
 --data_config ./conf/data/norm_layer6.yaml \
 --test

