import torch

checkpoint = torch.load(
    "./dinov2_results/best_linear_probe.pth",
    map_location="cpu",
    weights_only=True
)

print("저장된 항목:")
print(checkpoint.keys())

for key, value in checkpoint.items():
    if isinstance(value, dict):
        print(key, ":", list(value.keys())[:10])
    else:
        print(key, ":", value)