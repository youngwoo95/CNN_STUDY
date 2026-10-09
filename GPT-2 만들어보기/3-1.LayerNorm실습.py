import torch
import torch.nn as nn

x = torch.tensor([
    [10.0, 20.0, 30.0, 40.0],
    [100.0, 200.0, 300.0, 400.0]
])

norm = nn.LayerNorm(4)

output = norm(x)

print(output)
print(output.mean(dim=-1))
print(output.var(dim=-1, unbiased=False))