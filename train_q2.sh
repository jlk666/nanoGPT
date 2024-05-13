#!/bin/bash

csv_file="losses_data_q1.csv"

# Check if the CSV file exists
if [ -f "$csv_file" ]; then
    # If the file exists, remove it
    rm "$csv_file"
    echo "Existing $csv_file removed."
else
    # If the file doesn't exist, print a message
    echo "$csv_file not found. Skipping removal."
fi

# Define the base command and parameters that do not change
base_command="python train.py config/train_shakespeare_char.py"
base_params="--device=mps --compile=False --eval_iters=250 --log_interval=10 --block_size=64 --batch_size=12 --n_layer=4 --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0 --sliding_window=False --sliding_window_size=9999 --revisedMLP=False --register_token=0 --question_number=1"

# Define the number of heads
n_head=6

# Define the ratios you want to test
declare -a ratios=(8 32 64)

# Loop over each ratio
for ratio in "${ratios[@]}"; do
    # Calculate n_embd based on the ratio and n_head
    n_embd=384

    # Construct the full command with dynamic n_embd and n_head
    full_command="$base_command $base_params --n_head=$n_head --n_embd=$n_embd --interest_ratio=$ratio"

    # Echo command to terminal (optional, for visibility)
    echo "Running training with n_head=$n_head and n_embd=$n_embd (Ratio=$ratio)"
    echo $full_command

    # Execute the command
    eval $full_command
done

echo "All training runs completed."
