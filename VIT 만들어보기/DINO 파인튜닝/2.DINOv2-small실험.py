"""DINOv2-Small frozen-feature linear probe for insects_10.

Run from the project root containing insects_10/.
"""
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from transformers import AutoModel

DATA_DIR = Path('./insects_10')
OUTPUT_DIR = Path('./dinov2_results')
MODEL_ID = 'facebook/dinov2-small'
SEED = 42
FEATURE_BATCH_SIZE = 32  # lower to 16 or 8 if CUDA memory runs out
HEAD_BATCH_SIZE = 128
NUM_WORKERS = 2
EPOCHS = 100
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 15
RUN_FINAL_TEST = False  # set True only when finalizing the model

DEVICE = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')

# DINOv2-Small processor defaults: shortest edge 256, center crop 224,
# ImageNet mean/std. Keep these transforms fixed for linear probing.
EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225)),
])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(split):
    dataset = ImageFolder(DATA_DIR / split, transform=EVAL_TRANSFORM)
    return dataset


@torch.inference_mode()
def extract_features(backbone, dataset, split):
    loader = DataLoader(dataset, batch_size=FEATURE_BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=(DEVICE.type == 'cuda'))
    feats, targets = [], []
    backbone.eval()
    for batch_idx, (images, labels) in enumerate(loader, start=1):
        images = images.to(DEVICE, non_blocking=(DEVICE.type == 'cuda'))
        outputs = backbone(pixel_values=images)
        # [B, 257, 384] -> CLS [B, 384] for 224px and patch14
        cls_features = outputs.last_hidden_state[:, 0, :]
        feats.append(cls_features.cpu().float())
        targets.append(labels)
        if batch_idx % 20 == 0 or batch_idx == len(loader):
            print(f'  {split}: {batch_idx}/{len(loader)} batches')
    return torch.cat(feats), torch.cat(targets)


def run_epoch(head, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    head.train(is_train)
    total_loss, total_correct, total = 0.0, 0, 0
    for features, labels in loader:
        features = features.to(DEVICE)
        labels = labels.to(DEVICE)
        with torch.set_grad_enabled(is_train):
            logits = head(features)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        n = labels.size(0)
        total_loss += loss.item() * n
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += n
    return total_loss / total, total_correct / total * 100


def main():
    set_seed(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print('Device:', DEVICE)

    train_ds = make_dataset('train')
    val_ds = make_dataset('val')
    test_ds = make_dataset('test')
    class_names = train_ds.classes
    if len(class_names) != 10:
        raise ValueError(f'Expected 10 classes, got {len(class_names)}')
    if class_names != val_ds.classes or class_names != test_ds.classes:
        raise ValueError('Class order differs between splits')
    print(f'Train={len(train_ds)} Val={len(val_ds)} Test={len(test_ds)}')
    print('Classes:', class_names)
    (OUTPUT_DIR / 'class_names.json').write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding='utf-8')

    print(f'Loading pretrained backbone: {MODEL_ID}')
    backbone = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
    backbone.requires_grad_(False)

    print('Extracting frozen DINOv2 features (one-time pass)...')
    train_x, train_y = extract_features(backbone, train_ds, 'train')
    val_x, val_y = extract_features(backbone, val_ds, 'val')
    print('Feature shapes:', tuple(train_x.shape), tuple(val_x.shape))
    feature_dim = train_x.shape[1]
    del backbone  # free GPU memory; only the head is trained from here
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

    train_loader = DataLoader(TensorDataset(train_x, train_y),
                              batch_size=HEAD_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y),
                            batch_size=HEAD_BATCH_SIZE, shuffle=False)
    head = nn.Linear(feature_dim, len(class_names)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(head.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)

    best_acc = -1.0
    best_loss = float('inf')
    best_epoch = 0
    no_improve = 0
    ckpt_path = OUTPUT_DIR / 'best_linear_probe.pth'
    rows = []
    print('\n===== Linear Probing training =====')
    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(head, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(head, val_loader, criterion)
        print(f'Epoch {epoch:03d}/{EPOCHS} | Train Loss: {train_loss:.4f} | '
              f'Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | '
              f'Val Acc: {val_acc:.2f}%')
        rows.append(dict(epoch=epoch, train_loss=train_loss, train_acc=train_acc,
                         val_loss=val_loss, val_acc=val_acc))
        improved = val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss)
        if improved:
            best_acc, best_loss, best_epoch = val_acc, val_loss, epoch
            no_improve = 0
            torch.save({'head_state_dict': head.state_dict(),
                        'model_id': MODEL_ID, 'class_names': class_names,
                        'feature_dim': feature_dim, 'image_size': 224}, ckpt_path)
            print('  -> Best head saved')
        else:
            no_improve += 1
        if no_improve >= PATIENCE:
            print(f'Early stopping after {PATIENCE} epochs without improvement')
            break

    with (OUTPUT_DIR / 'linear_probe_history.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Best validation: {best_acc:.2f}% (epoch {best_epoch}, loss {best_loss:.4f})')
    print('Saved to:', ckpt_path)

    if RUN_FINAL_TEST:
        print('Extracting test features for final evaluation...')
        backbone = AutoModel.from_pretrained(MODEL_ID).to(DEVICE).eval()
        backbone.requires_grad_(False)
        test_x, test_y = extract_features(backbone, test_ds, 'test')
        checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        head.load_state_dict(checkpoint['head_state_dict'])
        test_loader = DataLoader(TensorDataset(test_x, test_y),
                                 batch_size=HEAD_BATCH_SIZE, shuffle=False)
        test_loss, test_acc = run_epoch(head, test_loader, criterion)
        print(f'FINAL TEST | Loss: {test_loss:.4f} | Accuracy: {test_acc:.2f}%')
    else:
        print('Test evaluation skipped. Set RUN_FINAL_TEST=True for final evaluation.')


if __name__ == '__main__':
    main()
