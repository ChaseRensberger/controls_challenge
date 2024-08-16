from . import BaseController
import csv

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

    with open('pid_steer_data.csv', 'a') as file:
        writer = csv.writer(file)
        writer.writerow([state.v_ego, state.a_ego, target_lataccel, state.roll_lataccel, self.previous_steering_angle])

    output = self.p * error + self.i * self.error_integral + self.d * error_diff
    self.previous_steering_angle = output
    return output
