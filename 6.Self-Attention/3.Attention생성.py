import torch
import torch.nn as nn
import math

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

# 2. sqrt(d_k)로 나누기
scores = QKT / math.sqrt(d_model)

print("Scaled Score")
print(scores)

# 3. softmax
attention_weights = torch.softmax(scores, dim=-1)


print("Attention Weights")
print(attention_weights)



# 4. V와 곱하기
output = attention_weights @ V

print("Attention output")
print(output)

