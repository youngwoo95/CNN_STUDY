# 토크나이저 기본
'''
text = "hello 월드😊"
ids = [ord(char) for char in list(text)]
print(ids)
'''

#chars = ['h', 'e', 'l', 'l', 'o', ' ', '월', '드', '😊']
#print(''.join(chars))

# 문자 단위 토크나이저
class CharTokenizer:
    def encode(self, text):
        return [ord(char) for char in list(text)]

    def decode(self, ids):
        return ''.join([chr(id) for id in ids])

tokenizer = CharTokenizer()
text = "hello 월드😊"

# 인코딩
ids = tokenizer.encode(text)
print(ids)

# 디코딩
decoded = tokenizer.decode(ids)
print(decoded)