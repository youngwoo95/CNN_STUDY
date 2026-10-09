import torch
import torch.nn as nn

vocab_size = 100
d_model = 4
max_seq_len = 128

token_ids = torch.tensor([12, 45, 83])

# 1. Token Embedding 생성
token_embedding = nn.Embedding(len(token_ids), d_model)

# 2. Position Embedding 생성
position_embedding = nn.Embedding(max_seq_len, d_model)

# 3. Position ID 생성
positions = torch.arange(len(token_ids))


# 4. Token Embedding 결과 계산
print(token_embedding)

# 5. Position Embedding 결과 계산
print(position_embedding)

# 6. 두 결과 더하기


# 7. 결과 Shape 출력