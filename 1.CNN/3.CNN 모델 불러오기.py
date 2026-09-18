# pip install torch
# pip install datasets huggingface_hub

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 학습할 때 사용했던 것과 동일해야 함
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),  # [3, 32, 32], 값 0~1
])

# 1. 모델 구조 생성
model = MyCNN(in_features=64 * 8 * 8, out_features=2).to(device)

# 2. 저장된 Weight 불러오기
model.load_state_dict(
    torch.load(
        "best_model.pt",
        map_location=device
    )
)

# 3. 평가 모드
model.eval()

@torch.no_grad()
def predict_image(image_path):

    # 이미지 읽기
    image = Image.open(image_path).convert("RGB")

    # Resize + Tensor
    image = transform(image)

    print("transform 후:", image.shape)

    # Batch 차원 추가
    image = image.unsqueeze(0)

    print("batch 추가 후: ", image.shape)

    # GPU로 이동
    image = image.to(device)

    # 모델 추론
    outputs = model(image)

    print("model output:", outputs)

    # 점수 → 확률
    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    print("probabilities:", probabilities)

    # 가장 높은 클래스
    predicted = probabilities.argmax(dim=1).item()

    cat_probability = probabilities[0][0].item() * 100
    dog_probability = probabilities[0][1].item() * 100

    classes = [
        "Cat",
        "Dog"
    ]

    print("----------------------")

    print(
        f"Prediction: {classes[predicted]}"
    )

    print(
        f"Cat: {cat_probability:.2f}%"
    )

    print(
        f"Dog: {dog_probability:.2f}%"
    )


predict_image("cat.jpg")

