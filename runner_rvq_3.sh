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

python train_vq.py \
 --data_config ./conf/data/norm_layer6.yaml \
 --previous_model_path ./runs/residualvq_ckpt1e3_codebookdim512_layer6_epo10/latest_checkpoint.pt \
 --ckpt_dir ./runs/residualvq_ckpt1e3_codebookdim512_layer6_epo2_continue_lr1e4_epoch2 \
 --model_config conf/models/residualvq.yaml \
 --lr 1e-4 \
 --train_epochs 2 \
#  --data_config \
#  --model_config \

# python train_vq.py --ckpt_dir ./runs/residualvq --model_config conf/models/residualvq.yaml --test