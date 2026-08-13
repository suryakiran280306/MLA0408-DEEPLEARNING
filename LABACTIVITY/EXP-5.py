import numpy as np
# Input values
x = np.array([-2, -1, 0, 1, 2])
# Weight and Bias
w = 2
b = 1
# Weighted Sum
z = w * x + b
# Sigmoid Function
def sigmoid(x):
return 1 / (1 + np.exp(-x))
# ReLU Function
def relu(x):
return np.maximum(0, x)
