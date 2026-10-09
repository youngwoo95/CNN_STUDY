import torch
import torch.nn as nn

class MyLayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()

        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # 1. 평균 계산
        mean = x.mean(dim=-1, keepdim=True)
        
        # 2. 분산 계산
        variance = x.var(
            dim = -1,
            unbiased=False,
            keepdim=True
        )

        # 3. 정규화
        normalize = (x-mean) / torch.sqrt(variance + self.eps)

        # 4. gamma, beta 적용
        x = (self.gamma * normalize) + self.beta

        return x


# 검증하기
x = torch.tensor([
    [10.0, 20.0, 30.0, 40.0],
    [100.0, 200.0, 300.0, 400.0]
])

my_norm = MyLayerNorm(d_model=4)
torch_norm = nn.LayerNorm(4)

my_output = my_norm(x)
torch_output = torch_norm(x)

print("직접 구현:")
print(my_output)

print("PyTorch 구현:")
print(torch_output)

print("결과가 거의 동일한가?")
print(torch.allclose(my_output, torch_output, atol=1e-5))