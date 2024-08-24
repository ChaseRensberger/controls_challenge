from . import BaseController
import numpy as np

# 10 * 20 * 10 = 2000

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,p,i,d):
    self.p = p
    self.i = i
    self.d = d
    self.error_integral = 0
    self.prev_error = 0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
      v_ego = state.v_ego # current velocity
      a_ego = state.a_ego # current acceleration
      roll_lataccel = state.roll_lataccel # current acceleration due to road roll
      next_target = future_plan[0].lataccel # next target lataccel

      error = (target_lataccel - current_lataccel)
      self.error_integral += error
      error_diff = error - self.prev_error
      self.prev_error = error
      return self.p * error + self.i * self.error_integral + self.d * error_diff