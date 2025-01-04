#!/bin/bash

set -x

RUN_NAME=0102_rvq_CB192_CBDIM1024_CBSIZE1024
MODEL_CONFIG_DIR=./conf/models/residualvq.yaml
DATA_CONFIG_DIR=./conf/data/example.yaml
OUTPUT_DIR=./exp/$RUN_NAME

mkdir -p $OUTPUT_DIR
cp $MODEL_CONFIG_DIR $OUTPUT_DIR/model_config.yaml
cp $DATA_CONFIG_DIR $OUTPUT_DIR/data_config.yaml
