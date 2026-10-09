import torch
import torch.nn as nn
import math

class MyLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # 1. 평균 계산
        mean = x.mean(dim=-1, keepdim=True)
        
        # 2. 분산 계산
        variance = x.var(
            dim = -1,
            unbiased=False,
            keepdim=True
        )

        # 3. 정규화
        normalize = (x-mean) / torch.sqrt(variance + self.eps)

        # 4. gamma, beta 적용
        x = (self.gamma * normalize) + self.beta

        return x

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

        # 5. Causal Masking
        mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device),
            diagonal=1
        ).bool()
        print("----")
        print(mask)

        scores = scores.masked_fill(mask, float("-inf"))

        # 6. Softmax
        attention = torch.softmax(scores, dim=-1)

        print("scores shape:", scores.shape)
        print("mask shape:", mask.shape)
        print("attention:", attention)

        # 7. V 가중합
        output = attention @ V

        # 8. Head 다시 합치기
        output = output.transpose(0, 1)

        output = output.reshape(
            seq_len,
            self.d_model
        )

        # 9. Output Projection
        output = self.W_o(output)

        return output


class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        '''
        특징 차원 확장
        '''
        # 1. Linear: d_model -> 4 * d_model
        self.linear1 = nn.Linear(d_model, 4 * d_model)

        '''
        비선형 활성화
        '''
        # 2. GELU 활성화 함수
        self.gelu = nn.GELU()

        '''
        특징 차원 축소
        '''
        # 3. Linear: 4 * d_model -> d_model
        self.linear2 = nn.Linear(4*d_model, d_model)

    def forward(self, x):
        # 위에서 만든 Layer들을 순서대로 적용
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

    
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.norm1 = MyLayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)

        self.norm2 = MyLayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x):
        residual = x

        # 1. LayerNorm 1 적용
        normalized = self.norm1(x)

        # 2. Multi-Head Attention 적용
        mhx = self.attention(normalized)

        # 3. Residual Connection
        # 원래 x + Attention 결과
        x = residual + normalized

        # 4. LayerNorm 2 적용
        x = self.norm2(x)

        # 5. FeedForward 적용
        x = self.ff(x)

        # 6. Residual Connection
        # 3번에서 구한 x + FeedForward 결과
        x = x+mhx

        return x


x = torch.randn(3, 4)

block = TransformerBlock(
    d_model=4,
    num_heads=2
)

output = block(x)

print(output.shape)