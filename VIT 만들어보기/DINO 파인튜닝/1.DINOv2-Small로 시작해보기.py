import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

MODEL_NAME = "facebook/dinov2-small"

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

backbone = AutoModel.from_pretrained(
    MODEL_NAME
)

# Backbone은 학습하지 않음
for param in backbone.parameters():
    param.requires_grad = False

class DINOInsectClassifier(nn.Module):
    def __init__(self, backbone, num_classes=10):
        super().__init__()

        self.backbone = backbone

        self.head = nn.Linear(
            backbone.config.hidden_size,
            num_classes
        )

    def forward(self, images):
        with torch.no_grad():
            features = self.backbone(
                pixel_values=images
            ).last_hidden_state

        cls_token = features[:, 0, :]

        return self.head(cls_token)

model = DINOInsectClassifier(
    backbone,
    num_classes=10
).cuda()

print(model)