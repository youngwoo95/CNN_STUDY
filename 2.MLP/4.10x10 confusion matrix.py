import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from sklearn.metrics import confusion_matrix, classification_report

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

    model = MyCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    early_stopping_patience = 6
    best_val_loss = float("inf")
    patience_count = 0


    for epoch in range(30):
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

            torch.save(model.state_dict(), "./CIFAR_best_model.pth")
            print('Best model 저장')
        else:
            patience_count += 1
            print(f"Validation 개선 없음 {patience_count}/{early_stopping_patience}")

        print(f"epoch {epoch + 1},\t"
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
    model.load_state_dict(torch.load("./CIFAR_best_model.pth",map_location=device))

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