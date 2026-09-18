# pip install torch
# pip install datasets huggingface_hub

import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.metrics import confusion_matrix, classification_report

# 이미지는 resize 필수.
'''
제각각 크기로는 사용못함.
'''
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # 공식 가중치
        std=[0.229, 0.224, 0.225] # 공식 가중치
    )
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # 공식 가중치
        std=[0.229, 0.224, 0.225] # 공식 가중치
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # 공식 가중치
        std=[0.229, 0.224, 0.225] # 공식 가중치
    )
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

def test_apply_transform(examples):
    examples["image"] = [
        test_transform(image.convert("RGB"))
        for image in examples["image"]
    ]
    return examples

@torch.no_grad()
def predict_dataset(model, data_loader, device):
    model.eval()

    y_true = []
    y_pred = []

    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(images)

        predicted = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predicted.cpu().tolist())
    return y_true, y_pred


@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for batch in data_loader:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        total_loss += loss.item()

        predicted = outputs.argmax(dim=1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    loss = total_loss / len(data_loader)
    accuracy = correct / total * 100

    return loss, accuracy


if __name__ == '__main__':
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("device:", device)

    ds = load_dataset("microsoft/cats_vs_dogs")

    # 1차 분할: Train 80%, 나머지 20%
    split_ds = ds["train"].train_test_split(
        test_size=0.2,
        seed=42,
        stratify_by_column = "labels"
    )

    # 분할해서 변수를 따로 저장한다.
    train_ds = split_ds["train"]
    temp_ds = split_ds["test"]

    # 2차 분할: 나머지 20%를 val / test 반반
    val_test_split = temp_ds.train_test_split(
        test_size=0.5,
        seed=42,
        stratify_by_column="labels"
    )

    val_ds = val_test_split["train"]
    test_ds = val_test_split["test"]

    # 그다음 둘다 transform 적용
    train_ds = train_ds.with_transform(train_apply_transform)
    val_ds = val_ds.with_transform(val_apply_transform)
    test_ds = test_ds.with_transform(test_apply_transform)

    # 그리고 DataLoader도 두 개 만든다.
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=2
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )


    weights = ResNet18_Weights.DEFAULT

    model = resnet18(weights=weights)

    # 1. 전체 Freeze
    for param in model.parameters():
        param.requires_grad = False

    # 2. layer4만 Freeze 해제
    for param in model.layer4.parameters():
        param.requires_grad = True

    # 3. 마지막 fc 교체
    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        2
    )

    # Feature Extractor에서 만든 Best Model 불러오기
    model.load_state_dict(
        torch.load(
            "best_model.pt",
            map_location=device
        )
    )

    model = model.to(device)

    criterion = nn.CrossEntropyLoss() # 분류 문제용 손실 함수

    '''
    filter()의 의미는
    model.parameter() 중에서 requires_grad = True인 것만 골라서 Optimizer에게 전달
    '''
    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )
    early_stopping_patience = 6

    baseline_val_loss, baseline_val_accuracy = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    print(
        f"Fine-Tuning 시작 전: "
        f"val loss {baseline_val_loss:.4f}, "
        f"val accuracy {baseline_val_accuracy:.2f}%"
    )

    best_val_loss = baseline_val_loss
    patience_count = 0

    # Fine-Tuning 전 Feature Extractor 상태를
    # 첫 번째 best 후보로 저장
    torch.save(
        model.state_dict(),
        "best_finetune_model.pt"
    )

    for epoch in range(50):

        current_lr = optimizer.param_groups[0]["lr"]

        model.train()
        model.bn1.eval()
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()

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

        # ====================
        # Best Model + Early Stopping
        # ===================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0


            torch.save(
                model.state_dict(),
                "best_finetune_model.pt"
            )
            print('Best model 저장 --------')
        else:
            patience_count += 1
            print(f"Validation 개선 없음 {patience_count}/{early_stopping_patience}")

        # ==================
        # Learning Rate Scheduler
        # =================
        scheduler.step(val_loss)

        # ==================
        # 결과 출력
        # ==================
        print(
            f"epoch {epoch + 1}, "
            f"lr {current_lr:.6f}, "
            f"train loss {train_loss:.4f}, "
            f"train accuracy {train_accuracy:.2f}%, "
            f"val loss {val_loss:.4f}, "
            f"val accuracy {val_accuracy:.2f}%"
        )

        # =========================
        # Early Stopping
        # =========================
        if patience_count >= early_stopping_patience:
            print(
                f"Validation 개선 없음 "
                f"{patience_count}/{early_stopping_patience}, "
                f"best val loss: {best_val_loss:.4f}"
            )
            break

    model.load_state_dict(
        torch.load(
             "best_finetune_model.pt",
            map_location=device
        )
    )

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    print(
        f"Test Loss: {test_loss:.4f}, "
        f"Test Accuracy: {test_accuracy:.2f}%"
    )

    y_true, y_pred = predict_dataset(
        model,
        test_loader,
        device
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    print("Confusion Matrix")
    print(cm)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Cat", "Dog"]
        )
    )