import numpy as np
from sklearn.metrics import mean_squared_error


a = [3, -0.5, 2, 7]
b = [2.5, 0.0, 2, 8]

def mse(a, b):
    mse = 0
    for i in range(len(a)):
        mse += (a[i] - b[i])**2
    return mse / len(a)

print(mean_squared_error(a, b))
print(mse(a, b))