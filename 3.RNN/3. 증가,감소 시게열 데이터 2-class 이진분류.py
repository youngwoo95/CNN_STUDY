import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 증가하는 데이터 1000개
up = torch.linspace(0, 1, 10).repeat(1000, 1)

# 감소하는 데이터 1000개
down = torch.linspace(1, 0,10).repeat(1000, 1)

# 약간의 노이즈 추가
up += torch.randn_like(up) * 0.05
down += torch.randn_like(down) * 0.05

# 합치기
x = torch.cat([up, down], dim=0)

# label
y = torch.cat([
    torch.zeros(1000, dtype=torch.long),
    torch.ones(1000, dtype=torch.long),
])

# RNN은 입력을 [Batch, Sequence, Feature] 형태로 받아야 하므로 차원 추가
x = x.unsqueeze(-1)

class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(input_size=1, hidden_size=16, batch_first=True)
        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        output, hidden = self.rnn(x)

        x = output[:, -1, :]
        x = self.fc(x)

        return x

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SimpleRNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)