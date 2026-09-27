import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.fc = nn.Linear(64 * 7 * 7, 10)  # 64채널, 7x7 이미지, 10개의 클래스

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        '''
        start_dim=1이 핵심이다.

        start_dim=1은
        output.shape이 현재 [64,64,7,7] 인데, 0번 차원인 배치 크기 64는 그대로 두고 1번 차원부터 끝까지 전부 펼치는 것이다.

        [64, 64, 7, 7]
                ↓
        [64, 64×7×7]
                ↓
        [64, 3136]
        '''
        flatten = torch.flatten(x, start_dim=1) # 분류기에 넣을 수 있도록 1차원으로 펼친다.
        x = self.fc(flatten)
        return x


model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


epochs = 5
for epoch in range(epochs):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # 예측 클래스
        '''
            argmax(dim=1)은 각 이미지마다 가장 점수가 높은 클래스 번호를 찾는 것이다.
            이미지 1 -> [0.2, 1.4, 0.1, ..., 3.8]
                                             ↑
                                           가장 큰 값 (해당 위치의 클래스 번호가 예측값이 됨)
        '''
        predicted = outputs.argmax(dim=1)
        # 맞힌 개수
        correct += (predicted == labels).sum().item()
        # 전체 이미지 개수
        total += labels.size(0)

    average_loss = total_loss / len(train_loader)
    train_accuracy = correct / total * 100

    print(
        f"Epoch {epoch + 1}/{epochs}, "
        f"Loss: {average_loss:.4f}, "
        f"Accuracy: {train_accuracy:.2f}%"
    )
