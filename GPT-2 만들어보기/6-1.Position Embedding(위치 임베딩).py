import torch
import torch.nn as nn

vocab_size = 100
d_model = 16
max_seq_len = 128

token_ids = torch.tensor([12, 45, 83, 21, 34])

#1. Token Embedding 생성
# (토큰 종류, 특징 개수)
token_embedding = nn.Embedding(vocab_size, d_model)

# 2. Position Embedding 생성
# 토큰 개수
position_embedding = nn.Embedding(max_seq_len, d_model)

# 3. position ID 생성
positions = torch.arange(len(token_ids))

# 4. 각 Embedding을 통해 벡터 생성
token_vectors = token_embedding(token_ids)
position_vectors = position_embedding(positions)

# 5. 두 벡터 더하기
x = token_vectors + position_vectors

print("Token IDs:", token_ids)
print("Position IDs:", positions)

print("Token Shape:", token_vectors.shape)
print("Position Shape:", position_vectors.shape)
print("Final Shape:", x.shape)