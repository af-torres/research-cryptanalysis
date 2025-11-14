#!/bin/bash

set -e

PYTHON_INTERPRETER=python
source .venv/bin/activate

# Check that the user provided a model version
if [ -z "$1" ]; then
    echo "Error: You must provide a model version as the first argument."
    echo "Usage: $0 <model_version>"
    exit 1
fi

MODEL_VERSION="$1"
input_noise=(.1 .2 .3 .4 .5 .6 .7 .8)
#input_noise=(.5 .6 .7 .8)

devices=(0 1 2 3)
total_devices=${#devices[@]}

for i in "${!input_noise[@]}"; do
    curr_dev=$((i % total_devices))
    "$PYTHON_INTERPRETER" auto_encoder_train.py --model_version="$MODEL_VERSION" --input_noise="${input_noise[$i]}" --device="$curr_dev" &
done

wait
