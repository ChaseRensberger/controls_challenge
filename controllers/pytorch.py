import torch
import torch.nn as nn
import torch.optim as optim
from . import BaseController
import os

class SimpleNN(nn.Module):
  def __init__(self):
    super(SimpleNN, self).__init__()
    self.layer1 = nn.Linear(4, 16)
    self.layer2 = nn.Linear(16, 1)

  def forward(self, x):
    x = torch.relu(self.layer1(x))
    x = self.layer2(x)
    return x

class Controller(BaseController):
  def __init__(self, model_path='model.pth'):
    self.model = SimpleNN()
    self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
    self.criterion = nn.MSELoss()

    self.model_path = model_path

    if os.path.exists(self.model_path):
        self.load_model()

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    inputs = torch.tensor([current_lataccel, state.roll_lataccel, state.v_ego, state.a_ego], dtype=torch.float32)
    target = torch.tensor([target_lataccel], dtype=torch.float32)

    prediction = self.model(inputs)
    loss = self.criterion(prediction, target)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    output = float(torch.clamp(prediction, -2, 2))
    self.save_model()
    return output
  
  def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.model_path)

  def load_model(self):
    checkpoint = torch.load(self.model_path, weights_only=True)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
