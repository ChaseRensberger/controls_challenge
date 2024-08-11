from tinyphysics import run_rollout
from controllers import custom
import numpy as np
import pandas as pd

# 0.3 0.09 -0.1
proportional_range = np.linspace(0.2, 0.4, 10)
integral_range = np.linspace(0.07, 0.11, 10)
derivative_range = np.linspace(-0.08, -0.12, 10)
data_path = "./data/00001.csv"
model_path = "./models/tinyphysics.onnx"

best_cost = 1000000000.0
best_p = 0
best_i = 0
best_d = 0

df = pd.read_csv(data_path)
num_rows = len(df)
print(f"Number of rows in 00000.csv: {num_rows}")
counter = 0

for p in proportional_range:
    for i in integral_range:
        for d in derivative_range:
            true_p = round(p, 3)
            true_i = round(i, 3)
            true_d = round(d, 3)
            counter += 1
            print(f"p: {true_p}, i: {true_i}, d: {true_d}, counter: {counter}")
            controller = custom.Controller(true_p, true_i, true_d)
            cost, _, _ = run_rollout(data_path, controller, model_path)
            if cost["total_cost"] < best_cost:
                best_cost = cost["total_cost"]
                best_p = true_p
                best_i = true_i
                best_d = true_d
                print(f"New best cost: {best_cost}, p: {best_p}, i: {best_i}, d: {best_d}")




