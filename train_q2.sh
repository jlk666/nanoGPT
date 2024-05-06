#!/bin/bash

# Define the base command and parameters that do not change
base_command="python train.py config/train_shakespeare_char.py"
base_params="--device=mps --compile=False --eval_iters=20 --log_interval=1 --block_size=64 --batch_size=12 --n_layer=4 --max_iters=5000 --lr_decay_iters=2000 --dropout=0.0"

# Define the number of heads
n_head=4

# Define the ratios you want to test
declare -a ratios=(64 32 8)

# Loop over each ratio
for ratio in "${ratios[@]}"; do
    # Calculate n_embd based on the ratio and n_head
    n_embd=$(($n_head * $ratio))

    # Construct the full command with dynamic n_embd and n_head
    full_command="$base_command $base_params --n_head=$n_head --n_embd=$n_embd"

    # Echo command to terminal (optional, for visibility)
    echo "Running training with n_head=$n_head and n_embd=$n_embd (Ratio=$ratio)"
    echo $full_command

    # Execute the command
    eval $full_command
done

echo "All training runs completed."
