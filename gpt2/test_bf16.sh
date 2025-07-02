#!/bin/bash

# Test script for bf16 training
echo "Testing GPT-2 training with bfloat16 precision..."

# Test with small config to verify bf16 works
python train.py \
    --dtype bfloat16 \
    --n_layer 2 \
    --n_head 2 \
    --n_embd 128 \
    --block_size 256 \
    --batch_size 2 \
    --max_iters 10 \
    --eval_interval 5 \
    --log_interval 1 \
    --compile False

echo "Bf16 test completed!"