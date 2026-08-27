import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate sample stock price data
data = np.array([
    100, 102, 101, 105, 110,
    108, 112, 115, 117, 120,
    118, 125, 130, 128, 135
])

# Create sequences
X = []
y = []

sequence_length = 3

for i in range(len(data) - sequence_length):
    X.append(data[i:i + sequence_length])
    y.append(data[i + sequence_length])

X = np.array(X)
y = np.array(y)

# Reshape data for LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Create LSTM model
model = Sequential()

model.add(
    LSTM(
        50,
        input_shape=(sequence_length, 1)
    )
)

model.add(Dense(1))

# Compile model
model.compile(
    optimizer='adam',
    loss='mean_squared_error'
)

# Train model
model.fit(
    X,
    y,
    epochs=100,
    verbose=1
)

# Prediction
prediction = model.predict(X)

print("Predicted Stock Prices:")
print(prediction)
