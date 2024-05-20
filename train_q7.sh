#!/bin/bash
csv_file="losses_data_q6.csv"

# Check if the CSV file exists
if [ -f "$csv_file" ]; then
    # If the file exists, remove it
    rm "$csv_file"
    echo "Existing $csv_file removed."
else
    # If the file doesn't exist, print a message
    echo "$csv_file not found. Skipping removal."
fi
echo "Testing with customized softmax"
    python train.py config/train_shakespeare_char.py --device=mps --compile=False --eval_iters=250 --log_interval=10 --block_size=100 --batch_size=12 --n_layer=4 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --interest_ratio=64 --sliding_window=False --sliding_window_size=0 --revisedMLP=False --register_token=0 --question_number=61


echo "Testing with register default model"
    python train.py config/train_shakespeare_char.py --device=mps --compile=False --eval_iters=250 --log_interval=10 --block_size=100 --batch_size=12 --n_layer=4 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --interest_ratio=64 --sliding_window=False --sliding_window_size=0 --revisedMLP=False --register_token=0 --question_number=62

#python q6_plot.py