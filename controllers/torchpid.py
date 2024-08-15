import torch
import torch.nn as nn
import torch.optim as optim
from . import BaseController

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
  def __init__(self):
    self.model = SimpleNN()
    self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)
    self.criterion = nn.MSELoss()
    self.p = 0.25
    self.i = 0.1
    self.d = -0.08
    self.error_integral = 0
    self.prev_error = 0
    self.max_difference_between_nn_and_pid = 0.02

  def update(self, target_lataccel, current_lataccel, state, future_plan):

    inputs = torch.tensor([current_lataccel, state.roll_lataccel, state.v_ego, state.a_ego], dtype=torch.float32)
    target = torch.tensor([target_lataccel], dtype=torch.float32)

    nn_pred = self.model(inputs)
    loss = self.criterion(nn_pred, target)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error

    pid_output = self.p * error + self.i * self.error_integral + self.d * error_diff
    float_nn_pred = float(nn_pred)

    if (abs(pid_output - float_nn_pred) > self.max_difference_between_nn_and_pid):
      return pid_output
    
    return float_nn_pred

