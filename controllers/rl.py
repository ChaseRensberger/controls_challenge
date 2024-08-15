import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from . import BaseController
import os

class DQNN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQNN, self).__init__()
        self.layer1 = nn.Linear(state_dim, 16)
        self.layer2 = nn.Linear(16, action_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
        return x

class Controller(BaseController):
    def __init__(self, model_path='model.pth', action_space=np.linspace(-2, 2, num=400), gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_dim = 4
        self.action_space = action_space
        self.action_dim = len(self.action_space)

        self.model = DQNN(self.state_dim, self.action_dim)
        self.target_model = DQNN(self.state_dim, self.action_dim)
        self.update_target_model()

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=2000)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.model_path = model_path
        if os.path.exists(self.model_path):
            self.load_model()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.choice(self.action_space)
        else:
            q_values = self.model(torch.tensor(state, dtype=torch.float32))
            return self.action_space[torch.argmax(q_values).item()]

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Prepare state and action
        inputs = [current_lataccel, state.roll_lataccel, state.v_ego, state.a_ego]
        state_tensor = torch.tensor(inputs, dtype=torch.float32)
        
        # Select action based on the current policy
        action = self.act(inputs)
        next_lataccel = self.simulate_next_state(state, action)  # You'll need to define this function
        
        # Reward calculation (e.g., based on the error between target and actual lateral acceleration)
        reward = -abs(target_lataccel - next_lataccel)
        
        # Prepare next state
        next_inputs = [next_lataccel, state.roll_lataccel, state.v_ego, state.a_ego]
        
        # Store experience in memory
        self.memory.append((inputs, action, reward, next_inputs))
        
        # Learn from a random sample in memory
        if len(self.memory) > 32:
            self.replay(32)
        
        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.save_model()
        
        # Return the action taken
        return action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state in minibatch:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            target = reward + self.gamma * torch.max(self.target_model(next_state_tensor)).item()
            target_f = self.model(state_tensor)
            target_f[list(self.action_space).index(action)] = target
            
            self.optimizer.zero_grad()
            loss = self.criterion(target_f, self.model(state_tensor))
            loss.backward()
            self.optimizer.step()
        
        # Periodically update the target model
        self.update_target_model()

    def save_model(self):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, self.model_path)

    def load_model(self):
        checkpoint = torch.load(self.model_path, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def simulate_next_state(self, state, action):
        # This is a placeholder function. Implement the logic to simulate the next state given the action.
        return state[0] + action  # Example implementation, modify as needed
