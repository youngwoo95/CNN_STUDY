import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from vit import VisionTransformer

DATA_DIR = Path("./insects_10")
OUTPUT_DIR = Path("./vit_results")
PRETRAINED_PATH = OUTPUT_DIR / "best_vit_finetuned.pth"
MODEL_PATH = OUTPUT_DIR / "best_vit_finetuned_90.pth"
HISTORY_PATH = OUTPUT_DIR / "finetune_61_90_history.csv"
START_EPOCH = 60
ADDITIONAL_EPOCHS = 30
PATIENCE = 10
LEARNING_RATE = 1.25e-5
WEIGHT_DECAY = 0.01
BATCH_SIZE = 32
NUM_WORKERS = 2
SEED = 42
IMAGE_SIZE = 128
PATCH_SIZE = 16
D_MODEL = 128
NUM_HEADS = 4
NUM_LAYERS = 3
# Keep Test completely out of the model-selection loop.
# Change to True only once you have finalized your model.
RUN_FINAL_TEST = False

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


train_transform = transforms.Compose([
    transforms.Resize((144, 144)),
    transforms.RandomCrop((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])
eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3),
])


def create_dataloaders():
    train_ds = ImageFolder(DATA_DIR / "train", transform=train_transform)
    val_ds = ImageFolder(DATA_DIR / "val", transform=eval_transform)
    test_ds = ImageFolder(DATA_DIR / "test", transform=eval_transform)
    class_names = train_ds.classes
    if len(class_names) != 10 or class_names != val_ds.classes or class_names != test_ds.classes:
        raise ValueError("Train/Val/Test 클래스 이름 및 순서를 확인하세요. 10종이어야 합니다.")
    def loader(ds, shuffle):
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle,
                          num_workers=NUM_WORKERS, pin_memory=device.type == "cuda")
    print(f"Train: {len(train_ds)} | Validation: {len(val_ds)} | Test: {len(test_ds)}")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False), test_ds, class_names


def create_model(num_classes):
    return VisionTransformer(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE,
                             d_model=D_MODEL, num_heads=NUM_HEADS,
                             num_layers=NUM_LAYERS, num_classes=num_classes).to(device)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum = correct = seen = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=device.type == "cuda")
        labels = labels.to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        n = labels.size(0)
        loss_sum += loss.item() * n
        correct += (logits.argmax(dim=1) == labels).sum().item()
        seen += n
    return loss_sum / seen, correct / seen * 100


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum = correct = seen = 0
    y_true, y_pred = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        pred = logits.argmax(dim=1)
        n = labels.size(0)
        loss_sum += loss.item() * n
        correct += (pred == labels).sum().item()
        seen += n
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    return loss_sum / seen, correct / seen * 100, y_true, y_pred


def save_history(history):
    df = pd.DataFrame(history)
    df.to_csv(HISTORY_PATH, index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df.epoch, df.train_loss, label="Train")
    axes[0].plot(df.epoch, df.val_loss, label="Validation")
    axes[0].set_title("Loss")
    axes[1].plot(df.epoch, df.train_acc, label="Train")
    axes[1].plot(df.epoch, df.val_acc, label="Validation")
    axes[1].set_title("Accuracy (%)")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "finetune_61_90_history.png", dpi=150)
    plt.close(fig)


def save_test_report(y_true, y_pred, class_names):
    labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels,
                                   target_names=class_names, digits=4,
                                   zero_division=0)
    print("\n===== Classification Report =====\n" + report)
    (OUTPUT_DIR / "finetune_61_90_classification_report.txt").write_text(report, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(11, 9))
    ConfusionMatrixDisplay(cm, display_labels=[str(x) for x in labels]).plot(
        ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Finetuned ViT: Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "finetune_61_90_confusion_matrix.png", dpi=150)
    plt.close(fig)


@torch.no_grad()
def predict_one(model, test_ds, class_names):
    model.eval()
    path, true_idx = test_ds.samples[0]
    with Image.open(path) as img:
        image = img.convert("RGB")
    x = eval_transform(image).unsqueeze(0).to(device)
    probs = torch.softmax(model(x), dim=1)[0]
    pred_idx = int(probs.argmax().item())
    print(f"샘플 실제: {class_names[true_idx]} | 예측: {class_names[pred_idx]} | 점수: {probs[pred_idx].item():.2%}")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(image)
    ax.set_title(f"True: {class_names[true_idx]}\nPred: {class_names[pred_idx]} ({probs[pred_idx].item():.2%})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "finetune_61_90_sample_prediction.png", dpi=150)
    plt.close(fig)


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PRETRAINED_PATH.is_file():
        raise FileNotFoundError(f"이전 체크포인트가 없습니다: {PRETRAINED_PATH}")
    print("사용 장치:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_loader, val_loader, test_loader, test_ds, class_names = create_dataloaders()
    model = create_model(len(class_names))
    model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device, weights_only=True))
    print("이전 60 Epoch 최고 가중치 로드 완료")
    (OUTPUT_DIR / "finetune_61_90_class_names.json").write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss, best_val_acc, _, _ = evaluate(model, val_loader, criterion)
    best_epoch = START_EPOCH  # Prior checkpoint selected on validation through epoch 60.
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"시작 Validation: Loss={best_val_loss:.4f} | Acc={best_val_acc:.2f}%")
    print("(시작 가중치는 이전 60 Epoch까지의 최고 모델입니다.)")
    no_improvement = 0
    history = []

    for step in range(1, ADDITIONAL_EPOCHS + 1):
        epoch = START_EPOCH + step
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc, lr=lr))
        print(f"Epoch {epoch:02d}/{START_EPOCH + ADDITIONAL_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr:.6f}")
        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_loss, best_val_acc = val_loss, val_acc
            best_epoch = epoch
            no_improvement = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print("  → 최고 성능 모델 저장")
        else:
            no_improvement += 1
        if no_improvement >= PATIENCE:
            print(f"Early Stopping: {PATIENCE} Epoch 동안 개선 없음")
            break

    save_history(history)
    print(f"학습 완료 | 최고 기록 시점: {'이전 체크포인트' if best_epoch == START_EPOCH else best_epoch} | "
          f"Val Acc: {best_val_acc:.2f}% | Val Loss: {best_val_loss:.4f}")
    print("최고 모델 저장:", MODEL_PATH)

    if RUN_FINAL_TEST:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion)
        print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
        save_test_report(y_true, y_pred, class_names)
        predict_one(model, test_ds, class_names)
    else:
        print("Test 평가를 건너뛰었습니다. 최종 선택 후 RUN_FINAL_TEST=True로 설정하세요.")


if __name__ == "__main__":
    main()
