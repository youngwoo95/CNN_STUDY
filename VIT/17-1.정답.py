import torch
import torch.nn as nn
import math


x = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [1.0, 1.0, 1.0, 0.0]
])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        seq_len = x.shape[0]

        # 1. Q, K, V 생성
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Head 분리
        Q = Q.view(seq_len, self.num_heads, self.d_head)
        K = K.view(seq_len, self.num_heads, self.d_head)
        V = V.view(seq_len, self.num_heads, self.d_head)

        # [seq_len, num_heads, d_head]
        #              ↓
        # [num_heads, seq_len, d_head]

        Q = Q.transpose(0, 1)
        K = K.transpose(0, 1)
        V = V.transpose(0, 1)

        # 3. Attention Score
        scores = Q @ K.transpose(-2, -1)

        # 4. Scaling
        scores = scores / math.sqrt(self.d_head)

        # 5. Softmax
        attention = torch.softmax(scores, dim=-1)

        # 6. V 가중합
        output = attention @ V

        # 7. Head 다시 합치기
        output = output.transpose(0, 1)

        output = output.reshape(
            seq_len,
            self.d_model
        )

        # 8. Output Projection
        output = self.W_o(output)

        return output


model = MultiHeadAttention(
    d_model=4,
    num_heads=2
)

output = model(x)

print(output)
print(output.shape)