import torch

def create_sliding_window_mask(matrix, k):
    n = matrix.size(0)
    mask = matrix.clone()
    
    for i in range(n):
        for j in range(n):
            if i - j > k:
                mask[i, j] = 0
    mask[mask == 0] = float('-inf')
    return mask


# Example usage:
attn_mask = torch.full((10, 10), float('inf'))
attn_mask = torch.tril(attn_mask.fill_(1))
sliding_window_mask = create_sliding_window_mask(attn_mask, 2)
print(sliding_window_mask)
