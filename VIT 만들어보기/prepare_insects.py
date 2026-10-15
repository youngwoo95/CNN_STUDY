
from collections import defaultdict
from pathlib import Path
import random
import shutil

import kagglehub
from PIL import Image

# --------------------------
# 설정
# --------------------------
SOURCE = Path(kagglehub.dataset_download(
    "kmldas/insect-identification-from-habitus-images"
))

OUTPUT = Path("./insects_10")
NUM_CLASSES = 10
IMAGES_PER_CLASS = 400
SEED = 42

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
rng = random.Random(SEED)

# --------------------------
# 1. 이미지 파일 찾기
# --------------------------
groups = defaultdict(list)

for path in SOURCE.rglob("*"):
    if path.is_file() and path.suffix.lower() in EXTENSIONS:
        groups[path.parent].append(path)

print("이미지가 들어 있는 폴더 수:", len(groups))

# --------------------------
# 2. 클래스 후보 확인
# --------------------------
candidates = sorted(
    [
        (folder, images)
        for folder, images in groups.items()
        if len(images) >= IMAGES_PER_CLASS
    ],
    key=lambda item: (-len(item[1]), str(item[0]))
)

print("\n클래스 후보:")
for folder, images in candidates[:15]:
    print(folder.relative_to(SOURCE), len(images))

if len(candidates) < NUM_CLASSES:
    raise RuntimeError(
        "이미지 400장 이상인 폴더가 10개 미만입니다. "
        "다운로드 파일 구조를 먼저 확인하세요."
    )

# --------------------------
# 3. 이미지 검증 후 10종 선정
# --------------------------
selected = []

for folder, images in candidates:
    valid_images = []

    for path in sorted(images):
        try:
            with Image.open(path) as img:
                img.verify()
            valid_images.append(path)
        except Exception:
            continue

    if len(valid_images) >= IMAGES_PER_CLASS:
        selected.append((folder, valid_images))

    if len(selected) == NUM_CLASSES:
        break

if len(selected) != NUM_CLASSES:
    raise RuntimeError("유효한 이미지가 있는 10개 클래스를 확보하지 못했습니다.")

if OUTPUT.exists():
    raise FileExistsError(
        f"{OUTPUT} 폴더가 이미 있습니다. "
        "기존 데이터를 확인하거나 다른 출력 경로를 사용하세요."
    )

# --------------------------
# 4. Train / Val / Test 분할
# --------------------------
for index, (folder, images) in enumerate(selected):
    class_name = f"class_{index:02d}_{folder.name}"

    images = images.copy()
    rng.shuffle(images)
    images = images[:IMAGES_PER_CLASS]

    splits = {
        "train": images[:280],
        "val": images[280:340],
        "test": images[340:400]
    }

    print(f"\n{class_name}")
    print("원본 이미지:", len(images))

    for split_name, paths in splits.items():
        target_dir = OUTPUT / split_name / class_name
        target_dir.mkdir(parents=True, exist_ok=True)

        for i, src in enumerate(paths):
            # 파일명이 충돌하지 않도록 번호 부여
            dst = target_dir / f"{i:04d}{src.suffix.lower()}"
            shutil.copy2(src, dst)

        print(f"  {split_name}: {len(paths)}")

print("\n데이터 준비 완료:", OUTPUT.resolve())
