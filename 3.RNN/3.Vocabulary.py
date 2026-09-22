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

# ==========================
# 2. 단어 빈도수 세기
# ==========================
for sample in train_data:
    tokens = tokenize(sample["text"])
    counter.update(tokens)

print(counter.most_common(5))

#==========================
# 3. 단어 사전 만들기
#==========================
vocab_size = 20000 # 상위 20,000개

vocab = {
    "<pad>":0,
    "<UNK>":1
}

for word, count in counter.most_common(vocab_size-2):
    vocab[word] = len(vocab)

print("vocab size:", len(vocab))

print(f'the:{vocab["the"]}')
print(f'movie:{vocab["movie"]}')

print(
    vocab.get(
        "asdasdasd",
        vocab["<UNK>"]
    )
)