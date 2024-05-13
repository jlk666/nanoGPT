#!/bin/bash

# Define register token values to test
register_tokens=(1 0)

# Iterate over each register token value
for token in "${register_tokens[@]}"
do
    echo "Testing with register token $token and revised MLP set to false"
    python train.py config/train_shakespeare_char.py --device=mps --compile=False --eval_iters=250 --log_interval=10 --block_size=100 --batch_size=12 --n_layer=4 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --interest_ratio=64 --sliding_window=False --sliding_window_size=9999 --revisedMLP=False --register_token=$token
done
