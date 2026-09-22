import re
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from datasets import load_dataset
from torch.utils.data import DataLoader
from collections import Counter

def tokenize(text):
    text = text.lower()
    text = re.sub(r"<br\s*/?>"," ",text)
    text = re.sub(r"[^a-z0-9']"," ",text)

    return text.split()

def encode_text(text, vocab, max_length):
    tokens = tokenize(text)
    token_ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens]
    
    # 너무 길면 자르기
    token_ids = token_ids[:max_length]

    # 짧으면 PAD 추가
    padding_length = max_length - len(token_ids)

    token_ids += [vocab["<PAD>"]] * padding_length

    return token_ids


class SentimentRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=64, hidden_size=32, num_classes=2):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        output, hidden = self.rnn(x)
        x = output[:, -1, :]
        x = self.fc(x)

        return x

class IMDbDataset(Dataset):
    def __init__(self, data, vocab, max_length):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        encoded = encode_text(sample["text"], self.vocab, self.max_length)
        x = torch.tensor(encoded, dtype=torch.long)
        y = torch.tensor(sample["label"], dtype=torch.long)
        return x, y


# 문장의 최대길이 지정
max_length = 200

dataset = load_dataset("stanfordnlp/imdb")

counter = Counter()

for sample in dataset["train"]:
    tokens = tokenize(sample["text"])
    counter.update(tokens)

vocab_size = 20000

vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for word, count in counter.most_common(vocab_size - 2):
    vocab[word] = len(vocab)

print("vocab size:", len(vocab))

train_dataset = IMDbDataset(
    dataset["train"],
    vocab,
    max_length
)

test_dataset = IMDbDataset(
    dataset["test"],
    vocab,
    max_length
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

x, y = next(iter(train_loader))

model = SentimentRNN(
    vocab_size=len(vocab)
)

logits = model(x)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = SentimentRNN(
    vocab_size=len(vocab)
).to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 5

for epoch in range(epochs):
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)

        loss = criterion(
            logits,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        pred = logits.argmax(dim=1)

        correct += (
            pred == y
        ).sum().item()

        total += y.size(0)

    accuracy = correct / total * 100

    print(
        f"Epoch {epoch + 1} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Accuracy: {accuracy:.2f}%"
    )



model.eval()

correct = 0
total = 0
test_loss = 0

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)

        loss = criterion(logits, y)

        test_loss += loss.item()

        pred = logits.argmax(dim=1)

        correct += (pred == y).sum().item()
        total += y.size(0)

test_accuracy = correct / total * 100

print(
    f"Test Loss: {test_loss / len(test_loader):.4f} | "
    f"Test Accuracy: {test_accuracy:.2f}%"
)