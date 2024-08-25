from . import BaseController
import numpy as np
import torch
import torch.nn as nn

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.leaky_relu = nn.LeakyReLU(0.001)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        x = self.tanh(x)
        return x * 2

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.model = SimpleNN()
    self.model.load_state_dict(torch.load('model.pt'))
    self.model.eval()

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    v_ego = state.v_ego # current velocity
    a_ego = state.a_ego # current acceleration
    roll_lataccel = state.roll_lataccel # current acceleration due to road roll

    X_test = torch.tensor([v_ego, a_ego, roll_lataccel, target_lataccel, current_lataccel], dtype=torch.float32)
    y_pred = self.model(X_test)
    print(y_pred.item())
    return y_pred.item()