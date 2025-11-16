import pandas as pd
import numpy as np
from pysr import PySRRegressor

df = pd.read_csv("datapoints.csv", header = 0)
data = df.iloc[:, 1 : 4].to_numpy()
target = df.iloc[:, 0].to_numpy()
X = data
y = target

model = PySRRegressor(
    maxsize=20,
    niterations=500,  # < Increase me for better results
    binary_operators=["+", "*", "/", "-"],
    unary_operators=[
        "exp",
        "inv(x) = 1/x",
        "square",
        "log"
        # ^ Custom operator (julia syntax)
    ],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    # ^ Define operator for SymPy as well
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
)

model.fit(X, y)

print(model)
