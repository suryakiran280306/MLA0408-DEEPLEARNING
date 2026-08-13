import numpy as np
import matplotlib.pyplot as plt
# Sample Dataset
X = np.array([1, 2, 3, 4, 5], dtype=float)

Y = np.array([2, 4, 6, 8, 10], dtype=float)
# Initialize parameters
w = 0
b = 0
learning_rate = 0.01
iterations = 100
n = len(X)
loss_history = []
# Gradient Descent
for i in range(iterations):
# Prediction
Y_pred = w * X + b
# Mean Squared Error Loss
loss = (1/n) * np.sum((Y - Y_pred) ** 2)
loss_history.append(loss)
# Gradients
dw = (-2/n) * np.sum(X * (Y - Y_pred))
db = (-2/n) * np.sum(Y - Y_pred)
# Update parameters
w = w - learning_rate * dw
b = b - learning_rate * db
# Final Parameters
print("Weight:", round(w,4))
print("Bias:", round(b,4))
print("Final Loss:", round(loss_history[-1],6))
# Plot Learning Curve
plt.plot(loss_history, color='blue')
plt.title("Learning Curve")
plt.xlabel("Iterations")

plt.ylabel("Loss (MSE)")
plt.grid(True)
plt.show()
