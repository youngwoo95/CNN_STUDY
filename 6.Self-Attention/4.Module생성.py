import torch
import torch.nn as nn
import math

x = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

d_model = 2

class SelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.d_model = d_model
        self.W_q = nn.Linear(d_model, d_model, False)
        self.W_k = nn.Linear(d_model, d_model, False)
        self.W_v = nn.Linear(d_model, d_model, False)

    def forward(self, x):

        # 1. Q, K, V 생성
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. QK^T
        #QKT = Q @ K.T # 해당 코드는 Batch가 들어가면 위험함.
        scores = Q @ K.transpose(-2, -1)

        # 3. Scaling
        scores = scores / math.sqrt(self.d_model)
        
        # 4. Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        
        # 5. V와 곱하기
        output = attention_weights @ V
        
        return output




