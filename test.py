from tinyphysics import run_rollout
from controllers import custom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 0.3 0.09 -0.1
# 0.358 0.1 0.03
# proportional_range = np.linspace(0.2, 0.4, 10)
# integral_range = np.linspace(0.07, 0.11, 10)
# derivative_range = np.linspace(-0.08, -0.12, 10)
# proportional_range = np.linspace(0, 0.6, 100)
# integral_range = np.linspace(0, 0.2, 100)
derivative_range = np.linspace(-0.2, 0.2, 100)
model_path = "./models/tinyphysics.onnx"
data_path = "./data/00001.csv"
# df = pd.read_csv(data_path)
# print(df.head())
# target_lat_accel = df["targetLateralAcceleration"]
# plt.plot(target_lat_accel)
# plt.xlabel("Time")
# plt.ylabel("Target Lateral Acceleration")
# plt.title("Future Target Lateral Acceleration")
# plt.show()

best_cost = 1000000000.0
best_p = 0.358
best_i = 0.01
best_d = 0

df = pd.read_csv(data_path)
num_rows = len(df)
print(f"Number of rows in 00000.csv: {num_rows}")
counter = 0

for d in derivative_range:
    true_d = round(d, 3)
    controller = custom.Controller(best_p, best_i, true_d)
    cost, _, _ = run_rollout(data_path, controller, model_path)
    if cost["total_cost"] < best_cost:
        best_cost = cost["total_cost"]
        best_d = true_d
        print(f"New best cost: {best_cost}, p: {best_p}, i: {best_i}, d: {best_d}")




