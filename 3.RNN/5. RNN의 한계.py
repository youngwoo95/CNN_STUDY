import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sequence_length = 100
num_samples = 2000

# 전체 시퀀스는 랜덤값
x = torch.randn(
    num_samples,
    sequence_length,
    1
)

# label은 0 또는 1
y = torch.randint(
    0,
    2,
    (num_samples,)
)

# 첫 번째 값에 정답을 심음
x[:, 0, 0] = y.float()


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

for epoch in range(20):
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


