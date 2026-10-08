import torch
import torch.nn as nn
import math

x = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0 ,0.0],
    [1.0, 1.0, 1.0, 0.0]
])


class SelfAttention(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()

        self.d_head = d_head

        self.W_q = nn.Linear(d_model, d_head, False)
        self.W_k = nn.Linear(d_model, d_head, False)
        self.W_v = nn.Linear(d_model, d_head, False)
        

    def forward(self, x):

        # 1. Q, K, V 생성
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. QK^T
        #QKT = Q @ K.T # 해당 코드는 Batch가 들어가면 위험함.
        scores = Q @ K.transpose(-2, -1)

        # 3. Scaling
        scores = scores / math.sqrt(self.d_head)
        
        # 4. Softmax
        attention_weights = torch.softmax(scores, dim=-1)

        
        # 5. V와 곱하기
        output = attention_weights @ V
        
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        # Head 하나당 사용할 차원
        self.d_head = d_model // num_heads

        # 여러 개의 Head 생성
        self.heads = nn.ModuleList([
            SelfAttention(d_model, self.d_head)
            for _ in range(num_heads)
        ])

        # Head들을 합친 후 최종 변환
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):

        # 각 Head에 같은 X 전달
        head_outputs = []

        for head in self.heads:
            out = head(x)
            head_outputs.append(out)

        # 마지막 차원 기준으로 연결
        multi_head_output = torch.cat(head_outputs, dim=-1)

        # 최종 Linear
        output = self.W_o(multi_head_output)

        return output

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ff_dim):
        super().__init__()

        # 1. Multi-Head Attention
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        # 2. LayerNorm 2개
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim,d_model)
        )

    def forward(self, x):

        # 1. Multi-Head Attention
        attention_output = self.attention(x)

        # 2. Residual + LayerNorm
        x1 = self.norm1(
            x + attention_output
        )

        # 3. Feed Forward
        ff_output = self.ffn(x1)

        # 4. Residual + LayerNorm
        x2 = self.norm2(x1 + ff_output)

        return x2

encoder = TransformerEncoderBlock(
    d_model=4,
    num_heads=2,
    ff_dim=8
)

output = encoder(x)

print(output)
print(output.shape)








