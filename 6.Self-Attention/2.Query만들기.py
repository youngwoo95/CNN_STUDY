import torch
import torch.nn as nn

x = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

d_model = 2

W_q = nn.Linear(
    in_features=d_model,
    out_features=d_model,
    bias=False
)

W_k = nn.Linear(
    in_features=d_model,
    out_features=d_model,
    bias=False
)

W_v = nn.Linear(
    in_features=d_model,
    out_features=d_model,
    bias=False
)

Q = W_q(x)
K = W_k(x)
V = W_v(x)

# 1. QK^T
QKT = Q @ K.T

