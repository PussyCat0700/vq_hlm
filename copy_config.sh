#!/bin/bash

set -x

RUN_NAME=0105_rvq_CN12_CD1k_CS1k_LR1E-5_EPOCH1_BS256
MODEL_CONFIG_DIR=./conf/models/residualvq.yaml
DATA_CONFIG_DIR=./conf/data/example.yaml
OUTPUT_DIR=./exp/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp $MODEL_CONFIG_DIR $OUTPUT_DIR/model_config.yaml
cp $DATA_CONFIG_DIR $OUTPUT_DIR/data_config.yaml
