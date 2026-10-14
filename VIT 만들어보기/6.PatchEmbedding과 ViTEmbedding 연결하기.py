import torch
import torch.nn as nn

# ========================
# 패치 임배딩
# ========================
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, d_model):
        super().__init__()

        self.patch_size = patch_size

        # 패치 하나의 RGB 데이터 개수
        # ex) 3 * 16 * 16 = 768
        patch_dim = 3 * patch_size * patch_size

        # 패치 벡터를 d_model 차원으로 변환
        # ex) 768 -> 128
        self.embedding = nn.Linear(patch_dim, d_model)

    def forward(self, x):
        p = self.patch_size

        # x.shape = [Batch, Channel, Height, Width]

        # 1. 세로 방향 패치 분할
        patches = x.unfold(2, p, p)

        # 2. 가로 방향 패치 분할
        patches = patches.unfold(3, p, p)

        # 3. 차원 순서 변경
        # [B, C, Ph, Pw, h, w]
        # -> [B, Ph, Pw, C, h, w]
        patches = patches.permute(0, 2, 3, 1, 4, 5)

        # 4. 패치 하나를 1차원 벡터로 변환
        # [B, Ph, Pw, C, h, w]
        # -> [B, Num_Patches, Patch_Dim]
        patches = patches.reshape(
            x.shape[0],
            -1,
            3 * p * p
        )

        # 5. Patch Embedding
        # [B, Num_Patches, Patch_Dim]
        # -> [B, Num_Patches, d_model]
        return self.embedding(patches)

# ========================
# CLS Token 임베딩
# ========================
class ViTEmbedding(nn.Module):
    def __init__(self, num_patches, d_model):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        # 위치 임베딩 생성
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, d_model)
        )

    def forward(self, patches):

        cls_tokens = self.cls_token.expand(
            patches.shape[0], -1, -1
        )

        # CLS Token 추가
        x = torch.cat([cls_tokens, patches], dim=1)

        # 위치 임베딩 추가
        x = x + self.pos_embedding

        return x

# 이미지 2장, RGB, 64x64
images = torch.randn(2, 3, 64, 64)

# 1. Patch Embedding
patch_embedding = PatchEmbedding(
    patch_size=16,
    d_model=128
)

patches = patch_embedding(images)

print(patches.shape)
# [2, 16, 128]

# 2. CLS Token + Positional Embedding
vit_embedding = ViTEmbedding(
    num_patches=16,
    d_model=128
)

x = vit_embedding(patches)




print(x.shape)