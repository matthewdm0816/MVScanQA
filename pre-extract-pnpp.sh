#!/bin/bash

# Auto GPU NUMBER
export SLURM_GPUS=$(($(echo $SLURM_JOB_GPUS | tr -cd , | wc -c)+1))
echo $SLURM_GPUS

export PORT=$(shuf -i 29000-30000 -n 1)
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_OFFLINE=1

# stdbuf -e0 -o0 python pre-extract-pnpp-feature.py
stdbuf -e0 -o0 python pre-extract-vote2cap-feature.py --feature_type box_features
