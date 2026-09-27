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


W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

Q = W_q(x)
K = W_k(x)
V = W_v(x)

# Head로 나눈다
Q = Q.reshape(x.size(0), num_heads, head_dim).transpose(0, 1)
K = K.reshape(x.size(0), num_heads, head_dim).transpose(0, 1)
V = V.reshape(x.size(0), num_heads, head_dim).transpose(0, 1)

print("Q:", Q.shape)
print("K:", K.shape)
print("V:", V.shape)

scores = Q @ K.transpose(-2, -1)
print("scores shape: ", scores.shape)