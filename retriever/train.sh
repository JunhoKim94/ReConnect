#!/bin/bash

echo "Running $PYTHON_TRAIN_SCRIPT"

DATA_DIR="./all_dataset"
TRAIN_PATH="$DATA_DIR/exp_train.jsonl"
DEV_PATH="$DATA_DIR/exp_dev.jsonl"


CUDA_VISIBLE_DEVICES=1 python "train.py" \
    --train_data_path "$TRAIN_PATH" \
    --dev_data_path "$DEV_PATH" 

