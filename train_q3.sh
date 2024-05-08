#!/bin/bash

python train.py config/train_shakespeare_char.py --device=mps --compile=False --eval_iters=20 --log_interval=1 --block_size=100 --batch_size=12 --n_layer=4 --n_head=6 --n_embd=384 --max_iters=1000 --lr_decay_iters=2000 --dropout=0.0 --interest_ratio=64 --sliding_window=True --sliding_window_size=100