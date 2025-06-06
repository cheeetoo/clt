#!/bin/bash

export TOKENIZERS_PARALLELISM=false

uv run torchrun \
    --nproc_per_node=8 \
    train.py \
    --model_name "Qwen/Qwen2.5-0.5B" \
    --dataset_name "HuggingFaceFW/fineweb" \
    --dataset_conf "" \
    --n_toks 300000000 \
    --bs 24 \
    --n_features 450048 \
    --bandwidth 1.0 \
    --threshold 0.03 \
    --lambda_p 3e-6 \
    --lambda_s 10.0 \
    --c 0.1 \
    --lr 2e-4 \
    --epochs 1 \
    --seq_len 128 \
    --out_path "model.pt"

if [ $? -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code $?"
    exit 1
fi
