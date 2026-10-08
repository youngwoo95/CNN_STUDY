#text = "I love cats. They are so cute.<|endoftext|>It is raining today. I need an umbrella.<|endoftext|>Apples are red. They taste sweets."
#texts = text.split("<|endoftext|>")
#print(texts)

# train_bpe() 함수 구현
# : 토큰 쌍의 출현 횟수를 센다.

from collections import defaultdict

# 기존버전
'''
def count_pairs(ids):
    counts = defaultdict(int)
    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts
'''

# 새버전
def count_pairs(ids, counts=None):
    if counts is None: # 새로 추가
        counts = defaultdict(int) # 새로 추가

    for pair in zip(ids, ids[1:]):
        counts[pair] += 1
    return counts

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

# 특수 토큰 처리 로직
def train_bpe(input_text, vocab_size, end_token="<|endoftext|>"):
    # 특수 토큰을 기준으로 분할
    texts = input_text.split(end_token)
    ids_list = [list(text.encode("utf-8")) for text in texts]

    # 기본 어휘(0~255) + 종료 토큰 1개를 제외한 수만큼 병합
    num_merges = vocab_size - 256 - 1
    print(num_merges)
    merge_rules = {}

    for step in range(num_merges):
        # 1) 모든 문서의 쌍 빈도를 합산
        counts = defaultdict(int)
        for ids in ids_list:
            count_pairs(ids, counts)

        if not counts:
            break

        # 2) 전체에서 가장 빈번한 쌍 하나 선택
        best_pair = max(counts, key=counts.get)
        new_id = 256 + step
        merge_rules[best_pair] = new_id

        # 3) 모든 문서에 병합 적용
        for i in range(len(ids_list)):
            ids_list[i] = merge(ids_list[i], best_pair, new_id)

    return merge_rules

# 개선된 train_bpe() 사용해보기
sample_text = "Hello World!<|endoftext|>This is BPE training."
print(train_bpe(sample_text, vocab_size=260))


