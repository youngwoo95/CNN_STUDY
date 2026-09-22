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

# 2. 단어 등장 횟수 계산
counter = Counter()

for sample in train_data:
    tokens = tokenize(sample["text"])
    counter.update(tokens)

print(counter.most_common(5)) # 자주 등장한 단어 TOP 5

# 3. Vocabulary 만들기
vocab_size = 20000

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word, count in counter.most_common(vocab_size - 2):
    vocab[word] = len(vocab)

# 문장의 최대길이 지정
max_length = 200

def encode_text(text, vocab, max_length):
    tokens = tokenize(text)
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    
    # 너무 길면 자르기
    token_ids = token_ids[:max_length]

    # 짧으면 PAD 추가
    padding_length = max_length - len(token_ids)

    token_ids += [vocab["<PAD>"]] * padding_length

    return token_ids


sample = dataset["train"][0]

encoded = encode_text(sample["text"],vocab,max_length)

print("label:", sample["label"])
print("encoded length:", len(encoded))
print(encoded[:30])

# decode
id_to_word = {
    idx: word
    for word, idx in vocab.items()
}

for token_id in encoded[:30]:
    print(token_id, "->", id_to_word[token_id])