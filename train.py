from tinyphysics import run_rollout
from controllers import pursuit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

num_files = 20000
offset = 100
data_path = "./data"
print("Loading data...")
all_data = [pd.read_csv(f"{data_path}/{str(i).zfill(5)}.csv") for i in range(num_files)]
print("Data loaded")
first_100_rows = [data.head(100) for data in all_data]
combined_data = pd.concat(first_100_rows)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 1)
        self.leaky_relu = nn.LeakyReLU(0.01)
        self.tanh = nn.Tanh()

        def forward(self, x):
            x = self.leaky_relu(self.fc1(x))
            x = self.leaky_relu(self.fc2(x))
            x = self.fc3(x)
            x = self.tanh(x)
            return x * 2
        
columns = ['vEgo', 'aEgo', 'roll', 'targetLateralAcceleration', 'steerCommand']
input_colums = ['aEgo', 'roll', 'vEgo', 'targetLateralAcceleration']
output_column = 'steerCommand'

model = SimpleNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

X_train = torch.tensor(combined_data[input_colums].values, dtype=torch.float32)
y_train = torch.tensor(combined_data[output_column].values, dtype=torch.float32)

epochs = 100
errors = []
print("Training model...")
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        error = loss.item()
        errors.append(error)
        print(f"Epoch {epoch}, Loss: {error}")

print("Saving model...")
torch.save(model.state_dict(), optimizer.state_dict(), "model.pth")
plt.plot(errors)
plt.show()



