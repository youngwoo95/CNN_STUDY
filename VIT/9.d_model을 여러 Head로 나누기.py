import torch
import torch.nn as nn
import math

x = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0]
])

seq_len = x.shape[0]
d_model = x.shape[1]
num_heads = 2

d_head = d_model // num_heads


print(f"d_model : {d_model}")
print(f"num_heads : {num_heads}")
print(f"d_head : {d_head}")