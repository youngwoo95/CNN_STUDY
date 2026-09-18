import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])

full_train_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform = train_transform
)
full_val_dataset = CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform = test_transform
)

generator = torch.Generator().manual_seed(42)
indices = torch.randperm(
    len(full_train_dataset),
    generator=generator
).tolist()

train_indices = indices[:45000]
val_indices = indices[45000:]

train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_val_dataset, val_indices)
test_dataset = CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=test_transform
)

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 4 * 4, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2 )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2 )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model = MyCNN().to(device)

    images, labels = next(iter(train_loader))

    images = images.to(device)
    labels = labels.to(device)

    outputs = model(images)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(5):
        model.train() # 학습 모드

        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device) # 이미지
            labels = labels.to(device) # 라벨

            outputs = model(images) # 모델에 넣고
            loss = criterion(outputs, labels) # 로스 게산하고

            optimizer.zero_grad() # 0으로 초기화하고
            loss.backward() # 역전파하고
            optimizer.step() # 업데이트한다

            total_loss += loss.item()

            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total * 100

        print(f"epoch {epoch + 1}, train loss {train_loss:.4f}, train accuracy {train_accuracy:.2f}%")