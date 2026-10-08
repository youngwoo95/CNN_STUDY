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


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()

        self.d_head = d_head

        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_q = nn.Linear(d_model, d_head, bias=False)
        self.W_k = nn.Linear(d_model, d_head, bias=False)
        self.W_v = nn.Linear(d_model, d_head, bias=False)

    def forward(self, x):
        # Q, K, V 생성
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Attention Score 계산
        scores = Q @ K.transpose(-2, -1)

        # Scaling
        scores = scores / math.sqrt( ㅜㅡ   )        

        # Softmax
        attention = torch.softmax(scores, dim=-1)

        # V를 가중합
        output = attention @ V
        return output



self_attention = SelfAttention(d_model, d_head)

output = self_attention(x)
print(output.shape)