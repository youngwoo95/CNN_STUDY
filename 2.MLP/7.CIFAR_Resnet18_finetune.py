import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from sklearn.metrics import confusion_matrix, classification_report
from torchvision.models import resnet18, ResNet18_Weights

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=16),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # 공개된 수치
        std=[0.229, 0.224, 0.225] # 공개된 수치
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # 공개된 수치
        std=[0.229, 0.224, 0.225] # 공개된 수치
    )
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

# ==============
# 검증
# ==============
@torch.no_grad()
def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item()

        predicted = outputs.argmax(dim=1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    val_loss = total_loss / len(data_loader)
    val_accuracy = correct / total * 100

    return val_loss, val_accuracy

# ==============================
# 에측 수집 함수
# =============================
@torch.no_grad()
def predict_dataset(model, data_loader, device):
    model.eval()

    y_true = [] # 실제 정답들
    y_pred = [] # 모델이 예측한 값들

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        predicted = outputs.argmax(dim=1)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(predicted.cpu().tolist())
    return y_true, y_pred



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2 )
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2 )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)

    # 전체 Backnbone Freeze
    for param in model.parameters():
        param.requires_grad = False

    # 마지막 layer4만 unfreeze
    for param in model.layer4.parameters():
        param.requires_grad = True

    # CIFAR-10 출력 10개
    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        10
    )

    # 처음부터 pretrained ImageNet weight만 쓰는 게 아니라,
    # 아까 학습한 Feature Extractor Best Model을 먼저 불러온다.
    model.load_state_dict(
        torch.load(
            "./CIFAR_resnet18_feature_best_model.pth",
            map_location=device
        )
    )

    # GPU 로 이동
    model = model.to(device)

    model.eval()  # 고정

    criterion = nn.CrossEntropyLoss()

    baseline_val_loss, baseline_val_accuracy = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    best_val_loss = baseline_val_loss
    patience_count = 0

    torch.save(
        model.state_dict(),
        "./CIFAR_resnet18_finetune_best_model.pth"
    )

    print(
        f"Fine-Tuning 시작 전: "
        f"val loss {baseline_val_loss:.4f}, "
        f"val accuracy {baseline_val_accuracy:.2f}%"
    )

    optimizer = torch.optim.Adam(
        filter(
            lambda p:p.requires_grad,
            model.parameters()
        ),
        lr=0.0001
    )

    # Scheduler 적용
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min",factor=0.5,patience=2)

    early_stopping_patience = 6

    # 학습 전에 어떤 레이어가 열렷는지 확인
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    for epoch in range(30):

        # 학습모드.
        model.train()

        # 앞쪽 레이어들 고정
        model.bn1.eval()
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()

        current_lr = optimizer.param_groups[0]["lr"]



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

            # 예측 클래스
            predicted = outputs.argmax(dim=1)
            # 맞힌 개수
            correct += (predicted == labels).sum().item()
            # 전체 이미지 개수
            total += labels.size(0)

        train_loss = total_loss / len(train_loader)
        train_accuracy = correct / total * 100

        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)


        # =================
        # Best Model + early stopping
        # ================
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0

            torch.save(
                model.state_dict(),
                "./CIFAR_resnet18_finetune_best_model.pth"
            )
            print('Best model 저장')
        else:
            patience_count += 1
            print(f"Validation 개선 없음 {patience_count}/{early_stopping_patience}")

        # ========================
        # Learning rate Scheduler
        # ========================
        scheduler.step(val_loss)

        print(f"epoch {epoch + 1},\t"
              f"lr {current_lr:.6f},\t"
              f"train loss {train_loss:.4f},\t"
              f"train accuracy {train_accuracy:.2f}%,\t"
              f"val loss {val_loss:.4f},\t"
              f"val accuracy {val_accuracy:.2f}%")

        # =======================
        # Early Stopping
        # =======================
        if patience_count >= early_stopping_patience:
            print(f"Validation 개선 없음,"
                  f"{patience_count}/{early_stopping_patience},"
                  f"best val loss {best_val_loss:.4f}")
            break

    # =============== 끝나는 지점

    # =====================
    # BEST Model 불러오기
    # =====================
    model.load_state_dict(
        torch.load(
            "./CIFAR_resnet18_finetune_best_model.pth",
            map_location=device
        )
    )

    # =====================
    # Test 평가
    # =====================
    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f},"
          f"Test Accuracy: {test_accuracy:.2f}%")

    # ====================
    # 10 x 10 Confusion Matrix
    # ====================
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]
    y_true, y_pred = predict_dataset(model, test_loader, device)
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix")
    print(cm)
    print(classification_report(y_true,y_pred,target_names=class_names))