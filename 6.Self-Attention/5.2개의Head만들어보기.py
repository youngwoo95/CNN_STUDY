import torch
import torch.nn as nn
import math

x = torch.tensor([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0 ,0.0],
    [1.0, 1.0, 1.0, 0.0]
])

d_model = 4
d_head = 2

class SelfAttention(nn.Module):
    def __init__(self, d_model, d_head):
        super().__init__()

        self.d_head = d_head

        self.W_q = nn.Linear(d_model, d_head, False)
        self.W_k = nn.Linear(d_model, d_head, False)
        self.W_v = nn.Linear(d_model, d_head, False)
        

    def forward(self, x):

        # 1. Q, K, V 생성
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. QK^T
        #QKT = Q @ K.T # 해당 코드는 Batch가 들어가면 위험함.
        scores = Q @ K.transpose(-2, -1)

        # 3. Scaling
        scores = scores / math.sqrt(self.d_head)
        
        # 4. Softmax
        attention_weights = torch.softmax(scores, dim=-1)

        
        # 5. V와 곱하기
        output = attention_weights @ V
        
        return output


head1 = SelfAttention(d_model, d_head)
head2 = SelfAttention(d_model, d_head)

out1 = head1(x)
out2 = head2(x)

multi_head_output = torch.cat(
    [out1, out2],
    dim=-1
)

W_o = nn.Linear(d_model, d_model)

output = W_o(multi_head_output)
print("head1:", out1.shape)
print("head2:", out2.shape)
print("concat:", multi_head_output.shape)
print("final:", output.shape)