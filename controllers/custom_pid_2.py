from . import BaseController
import numpy as np

class Controller(BaseController):
  """
  A simple PID controller with feedforward
  """
  def __init__(self,):
    self.p = 0.25
    self.i = 0.1
    self.d = -0.08
    self.ff = 0.1
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      pid_output = self.p * error + self.i * self.error_integral + self.d * error_diff
      feedforward_output = self.ff * target_lataccel
      return pid_output + feedforward_output