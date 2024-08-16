from . import BaseController
import csv
import os

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self,):
    self.p = 0.25
    self.i = 0.1
    self.d = -0.08
    self.error_integral = 0
    self.prev_error = 0

    self.previous_steering_angle = 0.0

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    error = (target_lataccel - current_lataccel)
    self.error_integral += error
    error_diff = error - self.prev_error
    self.prev_error = error

    file_exists = os.path.isfile('zero_steer_data.csv')
    if not file_exists:
        with open('zero_steer_data.csv', 'a') as file:
            writer = csv.writer(file)
            writer.writerow(['previous_steering_angle', 'current_lataccel', 'a_ego', 'v_ego', 'roll_lataccel', 'target_lataccel'])
    with open('zero_steer_data.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([self.previous_steering_angle, current_lataccel, state.a_ego, state.v_ego, state.roll_lataccel, target_lataccel])

    output = 0.0
    self.previous_steering_angle = output
    return output
    # return self.p * error + self.i * self.error_integral + self.d * error_diff
