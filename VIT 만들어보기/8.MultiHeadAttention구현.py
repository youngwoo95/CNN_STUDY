import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, N, D = x.shape

        # Q, K, V 생성
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Head 분리
        Q = Q.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(B, N, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(B, N, self.num_heads, self.d_head).transpose(1, 2)

        # Attention Score + Scaling
        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.d_head)

        # Softmax
        attention_weights = F.softmax(scores, dim=-1)

        # Attention 결과
        output = attention_weights @ V

        # Head 합치기
        output = output.transpose(1, 2)
        output = output.reshape(B, N, D)

        # Output Projection
        output = self.W_o(output)

        return output

# 이미지 4장, 토큰 17개, 특징 차원 128
x = torch.randn(4, 17, 128)

attention = MultiHeadAttention(
    d_model=128,
    num_heads=4
)

output = attention(x)

print(output.shape)