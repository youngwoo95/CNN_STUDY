import torch
import torch.nn as nn

class ViTEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, d_model)
        )

    def forward(self, patches):
        cls_tokens = self.cls_token.expand(
            patches.shape[0], -1, -1
        )

        result = torch.cat(
            [cls_tokens, patches],
            dim=1
        )
        return result

patches = torch.randn(3, 16, 128)
model = ViTEmbedding(d_model = 128)
result = model(patches)
print(result.shape)
