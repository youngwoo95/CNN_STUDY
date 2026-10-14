import torch

# 가상의 RGB 이미지 1장
# [batch, channel, height, wegit]
x = torch.randn(1, 3, 32, 32)

patch_size = 8

# 이미지를 패치 단위로 나누기
patches = x.unfold(2, patch_size, patch_size) # 보통 칸수, 넘김횟수를 갖게한다. 그래야 안 겹침
patches = patches.unfold(3, patch_size, patch_size) # 보통 칸수, 넘김횟수를 갖게한다. 그래야 안 겹침

print(f"unfold: {patches.shape}")