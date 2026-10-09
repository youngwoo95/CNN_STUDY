import torch
import torch.nn as nn
import math


# 토큰이 4개 있다고 가정
seq_len = 4

# 4 x 4 Attention Score
scores = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],
    [2.0, 3.0, 4.0, 5.0],
    [3.0, 4.0, 5.0, 6.0],
    [4.0, 5.0, 6.0, 7.0]
])

print(scores.shape)  # [4, 4]

# 1. Causal Mask 생성
mask = torch.triu(
    torch.ones(seq_len, seq_len),
    diagonal=1
)

print(mask)

# 2. Scores에 Mask 적용
masked_scores = scores.masked_fill(
    mask.bool(),
    float("-inf")
)

print(masked_scores)

softmax = torch.softmax(masked_scores, dim = -1)

print(f"softmax:{softmax}")