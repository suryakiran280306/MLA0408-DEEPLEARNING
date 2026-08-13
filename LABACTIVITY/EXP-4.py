import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# XOR Dataset
X = np.array([[0,0],

[0,1],
[1,0],
[1,1]], dtype=float)
y = np.array([[0],
[1],
[1],
[0]], dtype=float)
# Build Neural Network
model = Sequential([
Dense(4, input_dim=2, activation='relu'),
Dense(1, activation='sigmoid')
])
# Compile Model
model.compile(optimizer='adam',
loss='binary_crossentropy',
metrics=['accuracy'])
# Train Model
model.fit(X, y, epochs=500, verbose=0)
# Predictions
predictions = model.predict(X)
print("Predicted Outputs:")
print(np.round(predictions, 3))
