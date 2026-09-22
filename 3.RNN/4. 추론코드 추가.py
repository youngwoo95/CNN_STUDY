import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 증가하는 데이터 1000개
up = torch.linspace(0, 1, 10).repeat(1000, 1)

# 감소하는 데이터 1000개
down = torch.linspace(1, 0, 10).repeat(1000, 1)

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

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=16,
            batch_first=True)
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

dataset = TensorDataset(x, y)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        predicted = outputs.argmax(dim=1)

        correct += (predicted == labels).sum().item()

        total += labels.size(0)

    train_loss = total_loss / len(train_loader)
    train_accuracy = correct / total * 100

    print(
        f"epoch {epoch + 1}, "
        f"loss {train_loss:.4f}, "
        f"accuracy {train_accuracy:.2f}%"
    )


model.eval()

test_x = torch.tensor([
    0.10, 0.20, 0.18, 0.30, 0.27,
    0.42, 0.39, 0.55, 0.60, 0.58
], dtype=torch.float32)

test_x = test_x.unsqueeze(0).unsqueeze(-1)
test_x = test_x.to(device)

with torch.no_grad():
    output = model(test_x)

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted = probabilities.argmax(dim=1)

print("output:", output)
print("probabilities:", probabilities)

print(
    f"증가 확률: "
    f"{probabilities[0][0].item() * 100:.2f}%"
)

print(
    f"감소 확률: "
    f"{probabilities[0][1].item() * 100:.2f}%"
)

print(
    "예측:",
    "증가" if predicted.item() == 0 else "감소"
)