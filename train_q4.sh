#!/bin/bash
csv_file="losses_data_q3.csv"

# Check if the CSV file exists
if [ -f "$csv_file" ]; then
    # If the file exists, remove it
    rm "$csv_file"
    echo "Existing $csv_file removed."
else
    # If the file doesn't exist, print a message
    echo "$csv_file not found. Skipping removal."
fi
# Define whether to use the revised MLP or not
use_revised_mlp=(False True)

# Iterate over each setting
for revised in "${use_revised_mlp[@]}"
do
    echo "Testing with revised MLP set to $revised"
    python train.py config/train_shakespeare_char.py --device=mps --compile=False --eval_iters=250 --log_interval=10 --block_size=100 --batch_size=12 --n_layer=4 --n_head=6 --n_embd=384 --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --interest_ratio=64 --sliding_window=False --sliding_window_size=9999 --revisedMLP=$revised --question_number=3
done
