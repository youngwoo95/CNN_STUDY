import re
from collections import Counter
from datasets import load_dataset

dataset = load_dataset("stanfordnlp/imdb")
train_data = dataset["train"]

# 1. Token화 하기.
def tokenize(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>"," ",text)
    text = re.sub(r"[^a-z0-9']"," ",text)

    return text.split()

counter = Counter()

for sample in train_data:
    tokens = tokenize(sample["text"])
    counter.update(tokens)

print(counter.most_common(5)) # 자주 등장한 단어