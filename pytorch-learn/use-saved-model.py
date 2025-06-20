import torch
import torch.nn as nn
import torch.optim as optim


model_loaded = nn.Sequential(
    nn.Linear(2, 1)  # Input: 2 numbers, Output: 1 number (sum)
)

model_loaded.load_state_dict(torch.load("pytorch-learn/data/model_weights_2.pth"))

model_loaded.eval()

test = torch.tensor([8.0, 3.0])  # 8 + 3 = 11
output = model_loaded(test)
print("Model says:", output.item())

test = torch.tensor([999.0, 1.0])  # 1 + 1 = 2
output = model_loaded(test)
print("Model says:", output.item())

test = torch.tensor([99.0, 10.0])  # 10 + 10 = 20
output = model_loaded(test)
print("Model says:", output.item())


