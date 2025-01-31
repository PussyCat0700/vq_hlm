#!/bin/bash

set -x

RUN_NAME=0110_rvq_CN24_CD1k_CS8k_LR1E-4_EPOCH100_BS256
MODEL_CONFIG_DIR=./conf/models/residualvq.yaml
DATA_CONFIG_DIR=./conf/data/example.yaml
OUTPUT_DIR=./exp/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp $MODEL_CONFIG_DIR $OUTPUT_DIR/model_config.yaml
cp $DATA_CONFIG_DIR $OUTPUT_DIR/data_config.yaml
