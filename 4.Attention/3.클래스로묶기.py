import torch
import torch.nn as nn
import math

x = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

d_model = 2

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = Q @ K.transpose(-2, -1)

        d_k = K.size(-1)
        scaled_scores = scores / math.sqrt(d_k)

        weights = torch.softmax(
            scaled_scores,
            dim=-1
        )

        output = weights @ V

        return output

attention = SelfAttention(d_model)

output = attention(x)

print(output)
print(output.shape)