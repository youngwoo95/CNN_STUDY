import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        '''
        특징 차원 확장
        '''
        # 1. Linear: d_model -> 4 * d_model
        self.linear1 = nn.Linear(d_model, 4 * d_model)

        '''
        비선형 활성화
        '''
        # 2. GELU 활성화 함수
        self.gelu = nn.GELU()

        '''
        특징 차원 축소
        '''
        # 3. Linear: 4 * d_model -> d_model
        self.linear2 = nn.Linear(4*d_model, d_model)

    def forward(self, x):
        # 위에서 만든 Layer들을 순서대로 적용
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


x = torch.randn(3, 4)
model = FeedForward(d_model = 4)

output = model(x)

print(output.shape)