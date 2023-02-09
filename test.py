import torch
import torch.nn.functional as F
input = torch.zeros((4096, 27))
weight = torch.ones((16, 27))
bias = torch.ones((16))
res = F.linear(input, weight, bias)
print(res)