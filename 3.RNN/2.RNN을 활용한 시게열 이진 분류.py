import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=16,
            batch_first=True
        )

        self.fc = nn.Linear(16, 2)

    def forward(self, x):
        output, hidden = self.rnn(x)

        x = output[:, -1, :]
        x = self.fc(x)

        return x

model = SimpleRNN()
x = torch.randn(32, 5, 1)

outputs = model(x)

print(outputs.shape)