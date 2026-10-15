
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from vit import VisionTransformer


# ==========================================
# 1. 설정
# ==========================================
DATA_DIR = Path("./pollinator-insects-dataset")

IMAGE_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
NUM_CLASSES = 10

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("사용 장치:", device)


# ==========================================
# 2. Dataset
# ==========================================
class InsectDataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = Path(root)
        self.transform = transform

        csv_path = self.root / "data" / f"{split}.csv"
        self.df = pd.read_csv(csv_path)

        # 다운로드한 실제 이미지 경로 확인
        first = self.df.iloc[0]
        first_path = self.get_path(first)

        if not first_path.is_file():
            raise FileNotFoundError(
                f"이미지 파일을 찾지 못했습니다: {first_path}\n"
                "CSV의 image_path/image_filename과 "
                "실제 다운로드된 파일 구조를 확인하세요."
            )

    def get_path(self, row):
        # 이미지 파일명이 별도 필드로 존재하면 우선 사용
        if "image_filename" in self.df.columns:
            path = self.root / "images" / str(row["image_filename"])
            if path.is_file():
                return path

        raw_path = Path(str(row["image_path"]))

        if raw_path.is_absolute():
            return raw_path

        # CSV 경로가 저장소 루트 기준인 경우
        path = self.root / raw_path
        if path.is_file():
            return path

        # images 폴더 기준 상대 경로인 경우
        return self.root / "images" / raw_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = self.get_path(row)

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = int(row["label"])

        return image, label


# ==========================================
# 3. 이미지 전처리
# ==========================================
train_transform = transforms.Compose([
    transforms.Resize((144, 144)),

    transforms.RandomCrop((128, 128)),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(15),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ==========================================
# 4. DataLoader
# ==========================================
train_dataset = InsectDataset(
    DATA_DIR, "train", train_transform
)

print("클래스별 이미지 개수:")
print(train_dataset.df["label"].value_counts().sort_index())

print("실제 클래스 개수:")
print(train_dataset.df["label"].nunique())

print("라벨 목록:")
print(sorted(train_dataset.df["label"].unique()))


val_dataset = InsectDataset(
    DATA_DIR, "validation", val_transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    pin_memory=device.type == "cuda"
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=device.type == "cuda"
)

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))


# ==========================================
# 5. ViT 모델 생성
# ==========================================
model = VisionTransformer(
    image_size=IMAGE_SIZE,
    patch_size=16,
    d_model=128,
    num_heads=4,
    num_layers=3,
    num_classes=NUM_CLASSES
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.01
)


# ==========================================
# 6. 모델 학습
# ==========================================
def train_one_epoch():
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples * 100
    )


# ==========================================
# 7. Validation
# ==========================================
@torch.no_grad()
def validate():
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (
            logits.argmax(dim=1) == labels
        ).sum().item()
        total_samples += batch_size

    return (
        total_loss / total_samples,
        total_correct / total_samples * 100
    )


# ==========================================
# 8. 전체 학습 실행
# ==========================================
def main():
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch()
        val_loss, val_acc = validate()

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                "best_vit_insects_patch16_aug.pth"
            )

            print("  → 최고 성능 모델 저장")

    print(f"최고 Validation 정확도: {best_val_acc:.2f}%")

    # 검증
    test_dataset = InsectDataset(DATA_DIR, "test", val_transform )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False )

    model.load_state_dict(
        torch.load(
            "best_vit_insects_patch16_aug.pth",
            map_location=device,
            weights_only=True
        )
    )

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            predictions = logits.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    main()
