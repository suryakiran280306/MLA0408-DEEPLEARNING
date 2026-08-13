Python 3.10.0 (tags/v3.10.0:b494f59, Oct  4 2021, 19:00:18) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import numpy as np
# Cost Function: J(x) = x^2
def gradient_descent(lr):
x = 10
history = []
for i in range(20):
cost = x**2
history.append(cost)
gradient = 2*x
x = x - lr*gradient
return x, history
# Different Learning Rates
rates = [0.01, 0.1, 0.5]
for lr in rates:
x, history = gradient_descent(lr)
print("Learning Rate:", lr)
print("Final x:", round(x,4))
print("Final Cost:", round(history[-1],6))
print()