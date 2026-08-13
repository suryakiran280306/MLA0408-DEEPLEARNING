import numpy as np
# Inputs
x1 = 1

x2 = 2
# Weights
w1 = 0.5
w2 = 0.4
# Bias
b = 0.1
# Weighted Sum
z = x1*w1 + x2*w2 + b
# Sigmoid Function
output = 1/(1+np.exp(-z))
print("Weighted Sum =", z)
print("Neuron Output =", round(output,4))
