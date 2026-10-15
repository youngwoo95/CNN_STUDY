'''
from huggingface_hub import snapshot_download

dataset_dir = snapshot_download(
    repo_id="leonelgv/pollinator-insects-dataset",
    repo_type="dataset",
    local_dir="./pollinator-insects-dataset"
)

print("다운로드 위치:", dataset_dir)
'''
from pathlib import Path
import pandas as pd

data_dir = Path("./pollinator-insects-dataset/data")

for split in ["train", "validation", "test"]:
    df = pd.read_csv(data_dir / f"{split}.csv")

    print(f"\n[{split}]")
    print("이미지 수:", len(df))
    print("클래스 수:", df["label"].nunique())
    print("라벨 분포:")
    print(df["label"].value_counts().sort_index())