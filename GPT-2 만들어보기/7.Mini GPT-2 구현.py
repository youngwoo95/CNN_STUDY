import torch
import torch.nn as nn
import math

# ======================
# LayerNorm 구현
# ======================
class MyLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # 1. 평균 계싼
        mean = x.mean(dim=-1, keepdim=True)

        # 2. 분산 계산
        variance = x.var(
            dim = -1,
            unbiased = False,
            keepdim = True
        )

        # 3. 정규화
        normalize = (x-mean) / torch.sqrt(variance + self.eps)

        # 4. gamma, beta 적용
        x = (self.gamma * normalize) + self.beta
        return x

# ========================
# MultiHead Attention
# ========================
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

        scores = scores.masked_fill(mask, float("-inf"))

        # 6. softmax
        attention = torch.softmax(scores, dim=-1)

        # 7. V 가중합
        output = attention @ V

        # 8. Head 다시 합치기
        output = output.transpose(0, 1)
        output = output.reshape(seq_len, self.d_model)

        # 9. Output Projection
        output = self.W_o(output)

        return output

# ========================
# MultiHead Attention
# ========================
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        # 특징 차원 확장
        # 1. Linear : d_model -> 4 * d_model
        self.linear1 = nn.Linear(d_model, 4 * d_model)

        # 비선형 활성화
        # 2. GELU 활성화 함수
        self.gelu = nn.GELU()

        # 특징 차원 축소
        # 3. Linear : 4 * d_model -> d_model
        self.linear2 = nn.Linear(4*d_model, d_model)

    def forward(self, x):
        # 위에서 만든 Layer들을 순서대로 적용
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x

# ========================
# Transformer
# ========================
class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.norm1 = MyLayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, num_heads)

        self.norm2 = MyLayerNorm(d_model)
        self.ff = FeedForward(d_model)

    def forward(self, x):
        residual = x

        # 1. LayerNorm + Attention 적용
        normalized = self.norm1(x)
        mhx = self.attention(normalized)

        # 2. 첫 번째 Residual Connection
        # 원래 x + Attention 결과
        x = x + mhx

        # 3. LayerNorm + FeedForward
        normalized = self.norm2(x)
        ff_output = self.ff(normalized)

        # 4. 두 번째 Residual Connection
        x = x + ff_output
        return x

class MiniGPT2(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        max_seq_len,
        num_heads,
        num_layers
    ):
        super().__init__()

        # 1. Token Embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # 2. Position Embedding
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # 3. Transformer Block 여러 개 생성
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads)
            for _ in range(num_layers)
        ])

        # 4. 최종 LayerNorm
        self.final_norm = MyLayerNorm(d_model)

        # 5. 다음 토큰 점수 계산
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, token_ids):

        # 1. Position ID 생성
        positions = torch.arange(
            len(token_ids),
            device=token_ids.device
        )

        # 2. Token Embedding
        token_vectors = self.token_embedding(token_ids)

        # 3. Position Embedding
        position_vectors = self.position_embedding(positions)

        # 4. 두 Embedding 더하기
        x = token_vectors + position_vectors

        # 5. Transformer Block 통과
        for block in self.blocks:
            x = block(x)

        # 6. 최종 LayerNorm 적용
        x = self.final_norm(x)

        # 7. LM Head 통과
        logits = self.lm_head(x)

        return logits


model = MiniGPT2(
vocab_size=100,
d_model=16,
max_seq_len=128,
num_heads=4,
num_layers=2
)

# 2. 학습 데이터 생성
token_ids = torch.tensor([10, 20, 30, 40, 50, 60])

input_ids = token_ids[:-1]
target_ids = token_ids[1:]

# Loss 함수
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001
)

model.train()

for epoch in range(100):

    # 1. 이전 Gradient 초기화
    optimizer.zero_grad() # 0으로 초기화

    # 2. 모델에 input_ids 전달
    outputs = model(input_ids)

    # 3. Loss 계산
    loss = criterion(outputs, target_ids)

    # 4. 역전파
    loss.backward()

    # 5. 가중치 업데이트
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")


model.eval()

@torch.no_grad()
def generate(model, token_ids, max_new_tokens):
    model.eval()

    for _ in range(max_new_tokens):

        # 1. 현재 토큰으로 예측
        logits = model(token_ids)

        # 2. 마지막 위치의 점수
        next_logits = logits[-1]

        # 3. 가장 높은 점수의 토큰 선택
        next_token = torch.argmax(next_logits).view(1)

        # 4. 기존 토큰에 추가
        token_ids = torch.cat([token_ids, next_token])

    return token_ids


generated = generate(
    model,
    torch.tensor([10, 20]),
    max_new_tokens=4
)

print("생성 결과:", generated.tolist())