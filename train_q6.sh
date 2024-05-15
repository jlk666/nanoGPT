#!/bin/bash
echo "Testing with register token 1 + window_size 3"
    python train.py config/train_shakespeare_char.py --device=mps --compile=False --eval_iters=250 --log_interval=10 --block_size=100 --batch_size=12 --n_layer=4 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --interest_ratio=64 --sliding_window=True --sliding_window_size=3 --revisedMLP=False --register_token=1 --question_number=5


echo "Testing with register token 1"
    #python train.py config/train_shakespeare_char.py --device=mps --compile=False --eval_iters=250 --log_interval=10 --block_size=100 --batch_size=12 --n_layer=4 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --interest_ratio=64 --sliding_window=False --sliding_window_size=0 --revisedMLP=False --register_token=1 --question_number=45

echo "Testing with window_size 3"
``    #python train.py config/train_shakespeare_char.py --device=mps --compile=False --eval_iters=250 --log_interval=10 --block_size=100 --batch_size=12 --n_layer=4 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --interest_ratio=64 --sliding_window=True --sliding_window_size=3 --revisedMLP=False --register_token=0 --question_number=25

