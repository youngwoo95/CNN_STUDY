import torch
import torch.nn as nn

rnn = nn.RNN(
    input_size=10,
    hidden_size=20,
    batch_first=True
)

'''
32 = Batch Size
5 = Sequence length
10 = Feature Size
-> 즉 데이터 32개가 있고, 각 데이터는 5개의 순서를 가지며, 각 순서마다
숫자 10개가 들어있는 것이다.
'''
x = torch.randn(32, 5, 10)


output, hidden = rnn(x)

print(f"x shape: {x.shape}")
print(f"output shape: {output.shape}")
print(f"hidden shape: {hidden.shape}")
print(output[:, -1, :].shape)
print(torch.allclose(output[:, -1, :], hidden[0]))