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
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # [3, 32, 32], 값 0~1
])

def apply_transform(examples):
    examples["image"] = [
        transform(image.convert("RGB"))
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

        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features) # 선형회귀

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

if __name__ == '__main__':

    ds = load_dataset("microsoft/cats_vs_dogs")

    train_ds = ds["train"].with_transform(apply_transform)

    dataloader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)

    model = MyCNN(in_features=64 * 8 * 8, out_features=2)

    # -------
    batch = next(iter(dataloader))

    images = batch["image"]
    labels = batch["labels"]
    print(images.shape)
    print(labels.shape)

    output = model(images)
    print(output.shape)

    criterion = nn.CrossEntropyLoss() # 분류 문제용 손실 함수
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):

        model.train() # 학습 모드로 변경
        total_loss = 0
        correct = 0
        total = 0

        for batch in dataloader:
            images = batch["image"] # 이미지
            labels = batch["labels"] # 라벨

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

        avg_loss = total_loss / len(dataloader)
        accuracy = correct / total * 100

        print(
            f"epoch {epoch + 1}, "
            f"loss {avg_loss:.4f}, "
            f"accuracy {accuracy:.2f}%"
        )


'''
temp = ds["train"]
sample = temp[0]
img = sample["image"]      # PIL.Image 객체
label = sample["labels"]   # 0 = cat, 1 = dog
print(img.size, img.mode, label)
img.show()                 # OS 기본 이미지 뷰어로 열림
'''