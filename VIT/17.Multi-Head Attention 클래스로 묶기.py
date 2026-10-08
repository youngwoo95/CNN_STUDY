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

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)


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
        output = self.W_o(output)
        return output

model = MultiHeadAttention(d_model)
output = model(x)
print(output)
print(output.shape)