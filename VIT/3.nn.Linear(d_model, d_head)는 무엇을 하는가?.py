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
d_head = 2


# Q,K,V 만들기
W_q = nn.Linear(d_model, d_head, bias=False)
W_k = nn.Linear(d_model, d_head, bias=False)
W_v = nn.Linear(d_model, d_head, bias=False)


Q = W_q(x)
K = W_k(x)
V = W_v(x)

print(f"Q: {Q}")
print(f"K: {K}")
print(f"V: {V}")

print(Q.shape)
print(K.shape)
print(V.shape)

