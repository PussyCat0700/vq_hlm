#!/bin/bash

set -x

RUN_NAME=1221_rvq_CB4_CBDIM1024_BS256_EPOCH10
MODEL_CONFIG_DIR=./conf/models/residualvq.yaml
DATA_CONFIG_DIR=./conf/data/example.yaml
OUTPUT_DIR=./exp/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp $MODEL_CONFIG_DIR $OUTPUT_DIR/model_config.yaml
cp $DATA_CONFIG_DIR $OUTPUT_DIR/data_config.yaml

python train_vq.py --data_config $OUTPUT_DIR/data_config.yaml --model_config $OUTPUT_DIR/model_config.yaml --ckpt_dir $OUTPUT_DIR 2>&1 |tee $OUTPUT_DIR/train.log

# test
# python train_vq.py --data_config $OUTPUT_DIR/data_config.yaml --model_config $OUTPUT_DIR/model_config.yaml --ckpt_dir $OUTPUT_DIR --test 2>&1 |tee $OUTPUT_DIR/test.log
