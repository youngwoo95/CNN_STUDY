import torch

x = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0]
])

Q = x
K = x
V = x

scores = Q @ K.T


weights = torch.softmax(scores, dim=1)


output = weights @ V

print("scores")
print(scores)

print("weights")
print(weights)

print("output")
print(output)