'''
승인대기중
'''

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoModel

MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
DATA_DIR = Path("./insects_10")
RESULTS_DIR = Path("./dinov3_results")
CHECKPOINT = RESULTS_DIR / "best_linear_probe.pth"
SEED = 42
EXTRACT_BATCH_SIZE = 32
HEAD_BATCH_SIZE = 128
NUM_WORKERS = 2
EPOCHS = 100
PATIENCE = 15
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
RUN_FINAL_TEST = False  # Enable only for final evaluation.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Evaluation-style preprocessing, without stochastic augmentation.
# Match this to the prior DINOv2 script for the fairest comparison.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_datasets():
    train = ImageFolder(DATA_DIR / "train", transform=IMAGE_TRANSFORM)
    val = ImageFolder(DATA_DIR / "val", transform=IMAGE_TRANSFORM)
    test = ImageFolder(DATA_DIR / "test", transform=IMAGE_TRANSFORM)
    if len(train.classes) != 10 or train.classes != val.classes or train.classes != test.classes:
        raise ValueError("Expected exactly 10 identical class folders across train/val/test")
    return train, val, test


@torch.inference_mode()
def extract_features(backbone: nn.Module, dataset: ImageFolder, name: str):
    loader = DataLoader(dataset, batch_size=EXTRACT_BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=DEVICE.type == "cuda")
    backbone.eval()
    features, labels = [], []
    for i, (images, y) in enumerate(loader, 1):
        images = images.to(DEVICE, non_blocking=DEVICE.type == "cuda")
        outputs = backbone(pixel_values=images)
        # Official DINOv3 HF output includes pooler_output (CLS-style image representation).
        vector = outputs.pooler_output
        if vector is None:
            raise RuntimeError("DINOv3 pooler_output is missing; check transformers version")
        features.append(vector.float().cpu())
        labels.append(y.cpu())
        if i == 1 or i % 20 == 0 or i == len(loader):
            print(f"{name} features: {i}/{len(loader)} batches")
    return torch.cat(features), torch.cat(labels)


def head_evaluate(head, x, y, criterion):
    head.eval()
    with torch.inference_mode():
        logits = head(x.to(DEVICE))
        loss = criterion(logits, y.to(DEVICE)).item()
        accuracy = 100 * (logits.argmax(dim=1) == y.to(DEVICE)).float().mean().item()
        predictions = logits.argmax(dim=1).cpu()
    return loss, accuracy, predictions


def train_head(train_x, train_y, val_x, val_y, feature_dim, class_names):
    head = nn.Linear(feature_dim, len(class_names)).to(DEVICE)
    optimizer = torch.optim.AdamW(head.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    train_ds = TensorDataset(train_x, train_y)
    loader = DataLoader(train_ds, batch_size=HEAD_BATCH_SIZE, shuffle=True)
    best_acc, best_loss, best_epoch, stale = -1.0, float("inf"), 0, 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        head.train()
        loss_sum, correct, count = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = head(x)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += len(y)
        train_loss, train_acc = loss_sum / count, correct * 100 / count
        val_loss, val_acc, _ = head_evaluate(head, val_x, val_y, criterion)
        history.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                            val_loss=val_loss, val_acc=val_acc))
        print(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        if val_acc > best_acc + 1e-8 or (abs(val_acc - best_acc) <= 1e-8 and val_loss < best_loss):
            best_acc, best_loss, best_epoch, stale = val_acc, val_loss, epoch, 0
            torch.save({"head_state_dict": head.state_dict(), "model_id": MODEL_ID,
                        "feature_dim": feature_dim, "class_names": class_names,
                        "preprocess": "Resize(256), CenterCrop(224), ImageNet Normalize",
                        "best_epoch": epoch, "best_val_acc": val_acc, "best_val_loss": val_loss}, CHECKPOINT)
            print("  -> Best head saved")
        else:
            stale += 1
        if stale >= PATIENCE:
            print(f"Early stopping after {PATIENCE} epochs without improvement")
            break
    with (RESULTS_DIR / "training_history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    print(f"Best validation: {best_acc:.2f}% (epoch {best_epoch}, loss {best_loss:.4f})")
    print("Saved to:", CHECKPOINT)
    return head


def main():
    set_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Device:", DEVICE)
    train_ds, val_ds, test_ds = create_datasets()
    names = train_ds.classes
    print(f"Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)} Classes={len(names)}")
    (RESULTS_DIR / "class_names.json").write_text(json.dumps(names, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Loading pretrained backbone:", MODEL_ID)
    try:
        backbone = AutoModel.from_pretrained(MODEL_ID).to(DEVICE)
    except Exception as exc:
        raise RuntimeError("Cannot load DINOv3. Accept access terms on the HF model page, run `hf auth login`, "
                           "and upgrade transformers (`pip install -U transformers`).") from exc
    backbone.requires_grad_(False)
    backbone.eval()
    print("Extracting frozen DINOv3 features...")
    train_x, train_y = extract_features(backbone, train_ds, "Train")
    val_x, val_y = extract_features(backbone, val_ds, "Val")
    print("Feature shapes:", tuple(train_x.shape), tuple(val_x.shape))
    if train_x.shape[1] != val_x.shape[1]:
        raise ValueError("Feature dimension mismatch")
    del backbone
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    head = train_head(train_x, train_y, val_x, val_y, train_x.shape[1], names)
    if not RUN_FINAL_TEST:
        print("Test evaluation skipped. Set RUN_FINAL_TEST=True only after final model selection.")
        return
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    head.load_state_dict(checkpoint["head_state_dict"])
    print("Reloading backbone for one-time test evaluation...")
    backbone = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    backbone.requires_grad_(False)
    test_x, test_y = extract_features(backbone, test_ds, "Test")
    test_loss, test_acc, predictions = head_evaluate(head, test_x, test_y, nn.CrossEntropyLoss())
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(test_y.numpy(), predictions.numpy(),
                                   labels=list(range(len(names))), target_names=names, zero_division=0)
    print(report)
    (RESULTS_DIR / "test_classification_report.txt").write_text(report, encoding="utf-8")
    np.savetxt(RESULTS_DIR / "test_confusion_matrix.csv",
               confusion_matrix(test_y.numpy(), predictions.numpy(), labels=list(range(len(names)))),
               fmt="%d", delimiter=",")


if __name__ == "__main__":
    main()
