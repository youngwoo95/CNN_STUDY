import torch
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

print(train_dataset)
print(len(train_dataset))
print(len(test_dataset))

image, label = train_dataset[0] # 이건 한장의 데이터가 (image, label)로 나온다.
print(image.shape)
print(label)