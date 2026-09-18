# pip install torch
# pip install datasets huggingface_hub

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms


# 이미지는 resize 필수.
'''
제각각 크기로는 사용못함.
'''
train_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),  # [3, 32, 32], 값 0~1
])

val_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


def train_apply_transform(examples):
    examples["image"] = [
        train_transform(image.convert("RGB"))
        for image in examples["image"]
    ]
    return examples

def val_apply_transform(examples):
    examples["image"] = [
        val_transform(image.convert("RGB"))
        for image in examples["image"]
    ]
    return examples

class MyCNN(nn.Module):
    # 초기화 - 세팅 (갖다쓸 레이어들을 깔아둔다.)
    def __init__(self, in_features, out_features):
        super().__init__() # 이게 없으면 레이어 할당할때 에러남

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3,  padding=1) # 컨볼루션 레이어 1

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # 컨볼루션 레이어 2

        self.pool = nn.MaxPool2d(kernel_size=2) # 맥스 풀

        self.relu = nn.ReLU() # 렐루

        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features) # fully connect 레이어

    # 실행 순서
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for batch in val_loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        predicted = outputs.argmax(dim=1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    val_loss = total_loss / len(val_loader)
    val_accuracy = correct / total * 100

    return val_loss, val_accuracy


if __name__ == '__main__':
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)

    ds = load_dataset("microsoft/cats_vs_dogs")

    # 데이터 분할
    split_ds = ds["train"].train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column = "labels"
    )

    # 분할해서 변수를 따로 저장한다.
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]

    # 그다음 둘다 transform 적용
    train_ds = train_ds.with_transform(train_apply_transform)
    val_ds = val_ds.with_transform(val_apply_transform)

    # 그리고 DataLoader도 두 개 만든다.
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=32,
        shuffle=False,
        num_workers=2
    )


    model = MyCNN(in_features=64 * 8 * 8, out_features=2).to(device)

    # -------
    batch = next(iter(train_loader))

    images = batch["image"].to(device)
    labels = batch["labels"].to(device)
    print(images.shape)
    print(labels.shape)

    output = model(images)
    print(output.shape)

    criterion = nn.CrossEntropyLoss() # 분류 문제용 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_accuracy = 0.0

    for epoch in range(20):

        model.train() # 학습 모드로 변경
        total_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            images = batch["image"].to(device) # 이미지
            labels = batch["labels"].to(device) # 라벨

            outputs = model(images) # 모델에 넣고
            loss = criterion(outputs, labels) # 로스 계산하고

            optimizer.zero_grad() # 0으로 초기화하고
            loss.backward() # 역전파하고
            optimizer.step() # 업데이트한다. - range만큼 반복

            total_loss += loss.item()

            # 예측 클래스
            predicted = outputs.argmax(dim=1)
            # 맞힌 개수
            correct += (predicted == labels).sum().item()
            # 전체 이미지 개수
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total * 100

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                "best_model.pt"
            )

        print(
            f"epoch {epoch + 1}, "
            f"train loss {train_loss:.4f}, "
            f"train accuracy {train_accuracy:.2f}%, "
            f"val loss {val_loss:.4f}, "
            f"val accuracy {val_accuracy:.2f}%"
        )