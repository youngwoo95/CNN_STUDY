import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sequence_length = 10
num_samples = 2000

x = torch.zeros(
    num_samples,
    sequence_length,
    1
)

y = torch.randint(
    0,
    2,
    (num_samples,)
)

x[:, 0, 0] = y.float() * 2 - 1


class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=32,
            batch_first=True)
        self.fc = nn.Linear(32, 2)

          # Forget Gate Bias를 1로 초기화
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)

        x = output[:, -1, :]
        x = self.fc(x)

        return x

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SimpleLSTM().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

dataset = TensorDataset(x, y)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(50):
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


