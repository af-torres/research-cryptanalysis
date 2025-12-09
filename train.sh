#!/bin/bash

set -e

PYTHON_INTERPRETER=python
source .venv/bin/activate

#MODELS=(simple one_hot embedding reduced_embedding)
#MODELS=(reduced_by_char_embedding by_char_embedding)
MODELS=(lstm)
DATASET=random

devices=(0 1 2 3)
total_devices=${#devices[@]}

for i in "${!MODELS[@]}"; do
    curr_dev=$((i % total_devices))
    echo running training with --model_version="${MODELS[$i]}" --device="$curr_dev"
    "$PYTHON_INTERPRETER" auto_encoder_train.py \
        --model_version="${MODELS[$i]}" --device="$curr_dev" --dataset="$DATASET" &
done

wait
