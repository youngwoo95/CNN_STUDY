import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==================
# 멀티헤드 어텐션
# ==================
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

# =================
# FeedForward
# =================
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        hidden_dim = d_model * 4

        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.activation = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, d_model)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


# ===================
# 트랜스포머 엔코더
# ===================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, num_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model)

    def forward(self, x):

        # 1. Pre-LN + Multi-Head Attention
        x = x + self.attention(self.norm1(x))

        # 2. Pre-LN + FFN
        x = x + self.ffn(self.norm2(x))
        return x


# 테스트
x = torch.randn(4, 17, 128)

encoder = TransformerEncoderBlock(
    d_model=128,
    num_heads=4
)

output = encoder(x)

print(output.shape)