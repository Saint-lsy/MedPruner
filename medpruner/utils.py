import torch

def get_mask_entropy(scores,tau = 0.9):
    # Sort scores in descending order
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    
    # Cumulative sum
    cum_ratio = torch.cumsum(sorted_scores, 0)
    
    # Find first position where cumulative ratio >= tau
    if torch.any(cum_ratio >= tau):
        k = torch.argmax((cum_ratio >= tau).long()).item() + 1
    else:
        k = len(scores)

    return sorted_indices[:k]