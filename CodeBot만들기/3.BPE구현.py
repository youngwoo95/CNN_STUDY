from collections import defaultdict

# 토큰 쌍 횟수 구하기
def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts   

# 토큰 쌍 병합하기
def merge(ids, pair, new_id):
    merged_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) -1 and (ids[i], ids[i+1]) == pair:
            merged_ids.append(new_id)
            i += 2
        else:
            merged_ids.append(ids[i])
            i += 1

    return merged_ids

# BPE 학습
def train_bpe(text, vocab_size):
    # 텍스트를 0~255의 ID열로 변환
    ids = list(text.encode("utf-8"))

    # 1. 병합 횟수 결정
    num_merges = vocab_size - 256 # 256은 초기 어휘 크기
    merge_rules = {}

    for step in range(num_merges):
        # 인접 쌍의 출현 횟수 계산
        counts = count_pairs(ids)

        # 쌍이 존재하지 않으면 루프 종료
        if not counts:
            break
        
        # 가장 많이 등장한 쌍 선택
        best_pair = max(counts, key=counts.get)

        # 새로운 ID 생성
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # 토큰 쌍 병합
        ids = merge(ids, best_pair, new_id)

    return ids, merge_rules

# BPE를 적용한 인코딩과 디코딩
# BPE 토크나이저 구현
class BPETokenizer:
    def __init__(self, merge_rules):
        self.merge_rules = merge_rules
        
        # ID와 바이트열의 대응표 생성(0~255 등록)
        self.id_to_bytes = {i: bytes([i]) for i in range(256)}

        # 병합된 토큰은 원래 토큰의 바이트열을 연결해 생성
        for (id1, id2), new_id in merge_rules.items():
            self.id_to_bytes[new_id] = self.id_to_bytes[id1] + self.id_to_bytes[id2]

        # 어휘 크기 설정
        self.vocab_size = len(self.id_to_bytes)

    # 인코딩 구현
    def encode(self, text):
        ids = list(text.encode("utf-8"))

        # 학습할 때와 같은 순서로 병합 규칙 적용
        for merge_pair, new_id in self.merge_rules.items():
            ids = merge(ids, merge_pair, new_id)

        return ids
   
   # 디코딩 구현
   # decode() 메서드는 토큰 ID 열을 원래 텍스트로 되돌립니다.
    def decode(self, ids):
        # 1. 각 토큰 ID를 대응하는 바이트열로 변환
        byte_list = [self.id_to_bytes[i] for i in ids]

        # 2. 모든 바이트열을 연결
        text_bytes = b"".join(byte_list)

        # 3. 바이트열을 UTF-8 텍스트로 변환
        text = text_bytes.decode("utf-8", errors="replace") # 4.
        return text


# 동작 확인
# 학습된 병합 규칙
merge_rules = {(105, 115): 256, (256,32): 257,
                (105, 110): 258, (72, 101): 259}

# 토크나이저 생성
tokenizer = BPETokenizer(merge_rules)

# 텍스트 인코딩
text = "Hello월드😊"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print(ids)
print(decoded)