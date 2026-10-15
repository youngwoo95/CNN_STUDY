
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# 1. Patch Embedding
# ==========================================
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model):
        super().__init__()

        self.patch_size = patch_size

        patch_dim = 3 * patch_size * patch_size

        self.embedding = nn.Linear(
            patch_dim,
            d_model
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        p = self.patch_size

        # Patch 추출
        patches = x.unfold(2, p, p)
        patches = patches.unfold(3, p, p)

        # [B, H_patches, W_patches, 3, p, p]
        patches = patches.permute(0, 2, 3, 1, 4, 5)

        # [B, N, 3 * p * p]
        patches = patches.reshape(
            x.shape[0],
            -1,
            3 * p * p
        )

        # [B, N, d_model]
        return self.embedding(patches)


# ==========================================
# 2. CLS Token + Positional Embedding
# ==========================================
class ViTEmbedding(nn.Module):
    def __init__(self, num_patches, d_model):
        super().__init__()

        # CLS Token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        # Positional Embedding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, d_model)
        )

        nn.init.normal_(
            self.pos_embedding,
            mean=0.0,
            std=0.02
        )

    def forward(self, patches):
        # patches: [B, N, d_model]

        B = patches.shape[0]

        # CLS Token을 Batch 크기만큼 확장
        cls_tokens = self.cls_token.expand(
            B, -1, -1
        )

        # CLS Token 추가
        x = torch.cat(
            [cls_tokens, patches],
            dim=1
        )

        # 위치 정보 추가
        x = x + self.pos_embedding

        return x


# ==========================================
# 3. ViT Input
# ==========================================
class ViTInput(nn.Module):
    def __init__(self, image_size, patch_size, d_model):
        super().__init__()

        assert image_size % patch_size == 0

        num_patches = (image_size // patch_size) ** 2

        self.patch_embedding = PatchEmbedding(
            patch_size=patch_size,
            d_model=d_model
        )

        self.vit_embedding = ViTEmbedding(
            num_patches=num_patches,
            d_model=d_model
        )

    def forward(self, images):
        # [B, 3, H, W]
        patches = self.patch_embedding(images)

        # [B, N+1, d_model]
        x = self.vit_embedding(patches)

        return x


# ==========================================
# 4. Multi-Head Attention
# ==========================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q, K, V Projection
        self.W_q = nn.Linear(
            d_model, d_model, bias=False
        )

        self.W_k = nn.Linear(
            d_model, d_model, bias=False
        )

        self.W_v = nn.Linear(
            d_model, d_model, bias=False
        )

        # Output Projection
        self.W_o = nn.Linear(
            d_model, d_model, bias=False
        )

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape

        # 1. Q, K, V 생성
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Head 분리
        # [B, N, D] → [B, H, N, d_head]
        Q = Q.view(
            B, N, self.num_heads, self.d_head
        ).transpose(1, 2)

        K = K.view(
            B, N, self.num_heads, self.d_head
        ).transpose(1, 2)

        V = V.view(
            B, N, self.num_heads, self.d_head
        ).transpose(1, 2)

        # 3. Q × K^T
        scores = Q @ K.transpose(-2, -1)

        # 4. Scaling
        scores = scores / math.sqrt(self.d_head)

        # 5. Softmax
        attention_weights = F.softmax(
            scores,
            dim=-1
        )

        # 6. Attention × V
        output = attention_weights @ V

        # 7. Head 합치기
        # [B, H, N, d_head] → [B, N, D]
        output = output.transpose(1, 2)
        output = output.reshape(B, N, D)

        # 8. Output Projection
        output = self.W_o(output)

        return output


# ==========================================
# 5. Feed Forward Network
# ==========================================
class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        hidden_dim = d_model * 4

        self.fc1 = nn.Linear(
            d_model,
            hidden_dim
        )

        self.activation = nn.GELU()

        self.fc2 = nn.Linear(
            hidden_dim,
            d_model
        )

    def forward(self, x):
        # [B, N, D]
        x = self.fc1(x)

        # [B, N, 4*D]
        x = self.activation(x)

        # [B, N, D]
        x = self.fc2(x)

        return x


# ==========================================
# 6. Transformer Encoder Block (Pre-LN)
# ==========================================
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()

        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = FeedForward(d_model)

    def forward(self, x):
        # 1. LayerNorm → Attention → Residual
        x = x + self.attention(self.norm1(x))

        # 2. LayerNorm → FFN → Residual
        x = x + self.ffn(self.norm2(x))

        return x


# ==========================================
# 7. Transformer Encoder
# ==========================================
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, num_heads, num_layers):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                num_heads=num_heads
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


# ==========================================
# 8. Classification Head
# ==========================================
class ClassificationHead(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()

        self.fc = nn.Linear(
            d_model,
            num_classes
        )

    def forward(self, cls_token):
        return self.fc(cls_token)


# ==========================================
# 9. Vision Transformer
# ==========================================
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=64,
        patch_size=16,
        d_model=128,
        num_heads=4,
        num_layers=3,
        num_classes=10
    ):
        super().__init__()

        # 1. Patch + CLS + Position
        self.embedding = ViTInput(
            image_size=image_size,
            patch_size=patch_size,
            d_model=d_model
        )

        # 2. Transformer Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers
        )

        # Pre-LN Encoder 최종 정규화
        self.norm = nn.LayerNorm(d_model)

        # 3. Classification Head
        self.head = ClassificationHead(
            d_model=d_model,
            num_classes=num_classes
        )

    def forward(self, images):
        # [B, 3, 64, 64]

        # 1. Patch Embedding + CLS + Position
        x = self.embedding(images)
        # [B, 17, 128]

        # 2. Transformer Encoder
        x = self.encoder(x)
        # [B, 17, 128]

        # 3. 최종 LayerNorm
        x = self.norm(x)

        # 4. CLS Token 추출
        cls_token = x[:, 0, :]
        # [B, 128]

        # 5. Classification
        logits = self.head(cls_token)
        # [B, 10]

        return logits


# ==========================================
# 10. 모델 실행 테스트
# ==========================================
if __name__ == "__main__":

    torch.manual_seed(42)

    model = VisionTransformer(
        image_size=64,
        patch_size=16,
        d_model=128,
        num_heads=4,
        num_layers=3,
        num_classes=10
    )

    # RGB 이미지 8장
    images = torch.randn(8, 3, 64, 64)

    # Forward
    logits = model(images)

    print("입력 Shape:", images.shape)
    print("출력 Shape:", logits.shape)

    # 클래스별 확률
    probabilities = F.softmax(logits, dim=-1)

    # 예측 클래스
    predictions = logits.argmax(dim=-1)

    print("확률 Shape:", probabilities.shape)
    print("예측 클래스:", predictions)

    # 총 학습 파라미터 개수
    num_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"학습 파라미터: {num_params:,}")

    # 간단한 역전파 테스트
    labels = torch.randint(0, 10, (8,))

    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)

    loss.backward()

    print("Loss:", loss.item())
    print("역전파 성공!")
