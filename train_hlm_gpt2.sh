#!/bin/bash

set -x

RUN_NAME=$1
# RUN_NAME=0102_rvq_CB12_CBDIM1024_CBSIZE2048_EPOCH100
# MODEL_CONFIG_DIR=./conf/models/residualvq.yaml
# DATA_CONFIG_DIR=./conf/data/example.yaml
OUTPUT_DIR=./exp/$RUN_NAME

# mkdir -p $OUTPUT_DIR
# cp $MODEL_CONFIG_DIR $OUTPUT_DIR/model_config.yaml
# cp $DATA_CONFIG_DIR $OUTPUT_DIR/data_config.yaml

python train_hlm_gpt2.py \
| tee $OUTPUT_DIR/train.log