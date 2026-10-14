import torch
import torch.nn as nn

# 가상의 RGB 이미지 1장
# [batch, channel, height, wegit]
x = torch.randn(1, 3, 32, 32)

patch_size = 8

# 이미지를 패치 단위로 나누기
patches = x.unfold(2, patch_size, patch_size) # 보통 칸수, 넘김횟수를 갖게한다. 그래야 안 겹침
patches = patches.unfold(3, patch_size, patch_size) # 보통 칸수, 넘김횟수를 갖게한다. 그래야 안 겹침

print(f"unfold: {patches.shape}")
# torch.Size([1, 3, 4, 4, 8, 8])
#             B  C  Ph Pw H W
'''
B = 1 : 이미지 개수
C = 3 : 채널
Ph = 4 : 세로 패치 개수
Pw = 4 : 가로 패치 개수
H = 8 : 패치 높이
W = 8 : 패치 너비
'''

# 2. 차원 순서 변경
patches = patches.permute(0, 2, 3, 1, 4, 5)
print(f"permute : {patches.shape}")

# 3. 1차원 패치 벡터로 변환
patches = patches.reshape(1, 16, 192)
print(f"reshape : {patches}")


embedding = nn.Linear(192, 64)
patch_embeddings = embedding(patches)
print(patch_embeddings.shape)
