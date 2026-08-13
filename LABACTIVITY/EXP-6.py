from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
# -------------------------------
# Linearly Separable Dataset (AND)
# -------------------------------
X1 = [[0,0],
[0,1],
[1,0],
[1,1]]
y1 = [0,0,0,1]
model1 = Perceptron(max_iter=1000, random_state=42)
model1.fit(X1, y1)
pred1 = model1.predict(X1)
print("Linearly Separable Dataset")
print("Predictions:", pred1)
print("Accuracy:", accuracy_score(y1, pred1))
# -----------------------------------
# Non-Linearly Separable Dataset (XOR)
# -----------------------------------
X2 = [[0,0],
[0,1],
[1,0],
[1,1]]
y2 = [0,1,1,0]
model2 = Perceptron(max_iter=1000, random_state=42)
model2.fit(X2, y2)
pred2 = model2.predict(X2)
print("\nNon-Linearly Separable Dataset")
print("Predictions:", pred2)
print("Accuracy:", accuracy_score(y2, pred2))
