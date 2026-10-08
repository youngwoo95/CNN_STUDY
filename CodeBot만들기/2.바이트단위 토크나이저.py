class ByteTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))
    def decode(self, ids):
        return bytes(ids).decode("utf-8")

# 사용 예
tokenizer = ByteTokenizer()
text = "hello 월드😊"

# 인코딩
ids = tokenizer.encode(text)
print(ids)

# 디코딩
decoded = tokenizer.decode(ids)
print(decoded)