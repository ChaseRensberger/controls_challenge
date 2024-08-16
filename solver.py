import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer = nn.Linear(5, 1)

    def forward(self, x):
        x = self.layer(x)
        return x

model = SimpleNN()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


