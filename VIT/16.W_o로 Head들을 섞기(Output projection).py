import torch
import torch.nn as nn
import math

x = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0]
])

seq_len = x.shape[0] # 3
d_model = x.shape[1] # 4
num_heads = 2 # 2

d_head = d_model // num_heads # 2


W_q = nn.Linear(d_model, d_model, bias=False)
W_k = nn.Linear(d_model, d_model, bias=False)
W_v = nn.Linear(d_model, d_model, bias=False)

Q = W_q(x)
K = W_k(x)
V = W_v(x)


Q = Q.view(seq_len, num_heads, d_head)
K = K.view(seq_len, num_heads, d_head)
V = V.view(seq_len, num_heads, d_head)

Q = Q.transpose(0,1) # Q.shape = [2, 3, 2] Head개수 = 2, 토큰 개수 = 3, d_head = 2
K = K.transpose(0,1) # K.shape = [2, 3, 2] Head개수 = 2, 토큰 개수 = 3, d_head = 2
V = V.transpose(0,1) # V.shape = [2, 3, 2] Head개수 = 2, 토큰 개수 = 3, d_head = 2

scores = Q @ K.transpose(-2, -1)

scores = scores / math.sqrt(d_head)

attention = torch.softmax(scores, dim=-1)

output = attention @ V

output = output.transpose(0, 1)

output = output.reshape(seq_len, d_model)

W_o = nn.Linear(d_model, d_model, bias=False)
output = W_o(output)
print(output)
print(output.shape)