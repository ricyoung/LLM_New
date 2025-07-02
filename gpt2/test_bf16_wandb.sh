#!/bin/bash

# Test script for bf16 training with frequent W&B logging
echo "Testing GPT-2 training with bfloat16 and frequent W&B logging..."

./train --model medium --gpus 2 --steps 100 --batch-size 48 --dtype bfloat16 \
    --eval-interval 20 \
    --log-interval 5

echo "Bf16 W&B test completed!"