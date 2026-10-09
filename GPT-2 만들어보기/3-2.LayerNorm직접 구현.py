import torch


x = torch.tensor([
    [10.0, 20.0, 30.0, 40.0],
    [100.0, 200.0, 300.0, 400.0]
])

eps = 1e-5

# 1. 각 토큰의 평균 계산
mean = x.mean(dim=-1, keepdim=True)

# 2. 각 토큰의 분산 계산
variance = x.var(
    dim=-1,
    unbiased=False,
    keepdim=True
)


# 3. 정규화 (직접 작성)

normalized = (x - mean) / torch.sqrt(variance  + eps)

print(normalized)