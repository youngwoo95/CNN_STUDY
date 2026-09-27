import torch
import torch.nn as nn

x = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0, 1.0],
    [1.0, 1.0, 1.0, 1.0]
])

d_model = 4
num_heads = 2

head_dim = d_model // num_heads

print("x shape: ", x.shape)
print("head_dim: ", head_dim)

x_heads = x.reshape(x.size(0),num_heads,head_dim)
x_heads = x_heads.transpose(0, 1)
print(x_heads.shape)