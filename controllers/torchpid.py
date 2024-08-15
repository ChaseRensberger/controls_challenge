import torch
import torch.nn as nn
import torch.optim as optim
from . import BaseController
import os
import numpy as np

class SimpleNN(nn.Module):
  def __init__(self):
    super(SimpleNN, self).__init__()
    self.layer1 = nn.Linear(5, 16)
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
    self.p = 0.25
    self.i = 0.1
    self.d = -0.08
    self.error_integral = 0
    self.prev_error = 0

    self.model_path = model_path

    if os.path.exists(self.model_path):
        self.load_model()

  def update(self, target_lataccel, current_lataccel, state, future_plan):

    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error

    pid_output = self.p * error + self.i * self.error_integral + self.d * error_diff

    inputs = torch.tensor([pid_output, current_lataccel, state.roll_lataccel, state.v_ego, state.a_ego], dtype=torch.float32)
    target = torch.tensor([target_lataccel], dtype=torch.float32)

    nn_pred = self.model(inputs)
    loss = self.criterion(nn_pred, target)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.save_model()

    float_nn_pred = float(nn_pred)

    return np.clip(float_nn_pred, -2, 2)  
  
  def save_model(self):
      torch.save({
          'model_state_dict': self.model.state_dict(),
          'optimizer_state_dict': self.optimizer.state_dict(),
      }, self.model_path)

  def load_model(self):
    checkpoint = torch.load(self.model_path, weights_only=True)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

