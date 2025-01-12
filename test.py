import torch
import torch.optim as optim

# Define a model and an optimizer
model = torch.nn.Linear(10, 1)
optimizer = optim.SGD([
    {'params': model.parameters(), 'lr': 0.001},
    {'params': [model.bias], 'lr': 0.01}
])

# Accessing and printing optimizer param_groups
for param_group in optimizer.param_groups:
    print(param_group)