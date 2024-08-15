from . import BaseController
import numpy as np
from sklearn.neural_network import MLPRegressor

class Controller(BaseController):
  """
  A simple PID controller
  """
  def __init__(self):
    self.model = MLPRegressor(hidden_layer_sizes=(16,), activation='relu', solver='sgd', learning_rate_init=0.01, max_iter=1, warm_start=True)

  def update(self, target_lataccel, current_lataccel, state, future_plan):
    
    roll_lataccel = state.roll_lataccel 
    v_ego = state.v_ego
    a_ego = state.a_ego
    inputs = [[current_lataccel, roll_lataccel, v_ego, a_ego]]
    target = [target_lataccel]
    self.model.partial_fit(inputs, target)
    predicted_steering_angle = self.model.predict(inputs)[0]
    return np.clip(predicted_steering_angle, -2, 2)
    
