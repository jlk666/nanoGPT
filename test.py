import torch
import numpy as np

def create_sliding_window_mask(matrix, k):
    n = matrix.size(0)
    mask = matrix.clone()
    
    for i in range(n):
        for j in range(n):
            if i - j > k:
                mask[i, j] = 0
    mask[mask == 0] = float('-inf')
    return mask

def create_sliding_window_mask_registerToken(matrix, n):
    # Create an n x n matrix filled with zeros

    # Fill the first n columns with 1
    matrix[:, :n] = 1

    # Set elements above the diagonal to -inf
    matrix = torch.tril(matrix, diagonal=0)  # This sets elements above the diagonal to 0
    matrix[matrix == 0] = float('-inf')      # Set those 0s to -inf

    return matrix

# Example usage:
attn_mask = torch.full((8, 8), float('inf'))
attn_mask = torch.tril(attn_mask.fill_(1))
sliding_window_mask = create_sliding_window_mask(attn_mask, 3)
sliding_window_mask = create_sliding_window_mask_registerToken(sliding_window_mask,3) 
print(sliding_window_mask)

