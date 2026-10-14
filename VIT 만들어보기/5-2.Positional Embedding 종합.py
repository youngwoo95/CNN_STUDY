import torch
import torch.nn as nn

class ViTEmbedding(nn.Module):
    def __init__(self, num_patches, d_model):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, d_model)
        )

        nn.init.normal_(
            self.pos_embedding,
            mean=0.0,
            std=0.02
        )

    def forward(self, patches):
        cls_tokens = self.cls_token.expand(
            patches.shape[0], -1, -1
        )

        x = torch.cat([cls_tokens, patches], dim=1)

        x = x + self.pos_embedding
        return x

patches = torch.randn(5, 36, 96)

model = ViTEmbedding(
    num_patches=36,
    d_model=96
)
result = model(patches)
print(result.shape)