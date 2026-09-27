import torch
import torch.nn as nn
import math

x = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

d_model = 2

W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

Q = W_q(x)
K = W_k(x)
V = W_v(x)

print(f"Q : {Q}")
print(f"K : {K}")
print(f"V : {V}")

scores = Q @ K.T

d_k = K.size(-1)
print(d_k)
scaled_scores = scores / math.sqrt(d_k)

print("scaled scores:")
print(scaled_scores)

weights = torch.softmax(
    scaled_scores,
    dim=-1
)

print(weights)

output = weights @ V

print("output:")
print(output)
print("output shape:", output.shape)