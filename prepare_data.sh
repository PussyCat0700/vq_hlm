#!/bin/bash

set -x

RUN_NAME=$1
# RUN_NAME=0102_rvq_CB12_CBDIM1024_CBSIZE2048_EPOCH100
# MODEL_CONFIG_DIR=./conf/models/residualvq.yaml
# DATA_CONFIG_DIR=./conf/data/example.yaml
OUTPUT_DIR=./exp/$RUN_NAME

python prepare_hlm_token.py --data_config $OUTPUT_DIR/data_config.yaml --model_config $OUTPUT_DIR/model_config.yaml --ckpt_dir $OUTPUT_DIR
echo "Remenber to copy htoken.py to your data folder!"