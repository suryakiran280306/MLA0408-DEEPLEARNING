import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load stock dataset
data = pd.read_csv("stock_data.csv")

# Use closing price
prices = data["Close"].values.reshape(-1, 1)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# Create time sequences
X = []
y = []

sequence_length = 60

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

# Split training and testing data
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# Build LSTM model
model = Sequential([
    LSTM(
        64,
        return_sequences=True,
        input_shape=(X_train.shape[1], 1)
    ),

    Dropout(0.2),

    LSTM(64),

    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1)
])

# Compile model
model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

# Train model using BPTT automatically
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Predict
predictions = model.predict(X_test)

# Convert back to original price scale
predictions = scaler.inverse_transform(predictions)

print("Predicted Stock Prices:")
print(predictions[:10])
