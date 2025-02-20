#!/bin/bash

set -x

RUN_NAME=0220_CN24_CD1k_CS8k_LR1E-3_EPOCH100_BS256
OUTPUT_DIR=./exp/$RUN_NAME

# test
python train_vq.py --data_config $OUTPUT_DIR/data_config.yaml --model_config $OUTPUT_DIR/model_config.yaml --ckpt_dir $OUTPUT_DIR --test 2>&1 |tee $OUTPUT_DIR/test.log
