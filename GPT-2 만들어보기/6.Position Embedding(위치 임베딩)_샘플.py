import torch
import torch.nn as nn

seq_len = 3
d_model = 4
max_seq_len = 128

position_embedding = nn.Embedding(max_seq_len, d_model)

positions = torch.arange(seq_len)

print(positions)
print(position_embedding(positions).shape)