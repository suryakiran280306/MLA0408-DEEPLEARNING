import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Model
import numpy as np

# Load pre-trained ResNet50
cnn_model = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg'
)

# Freeze CNN layers
cnn_model.trainable = False


# Function to extract features from video frame
def extract_features(frame):

    frame = tf.image.resize(
        frame,
        (224, 224)
    )

    frame = np.expand_dims(frame, axis=0)

    frame = preprocess_input(frame)

    features = cnn_model.predict(frame)

    return features


# Caption generation model

feature_input = tf.keras.Input(shape=(2048,))

caption_input = tf.keras.Input(
    shape=(None,)
)

# Convert words into vectors
caption_embedding = tf.keras.layers.Embedding(
    input_dim=5000,
    output_dim=256
)(caption_input)

# LSTM Decoder
lstm_output = LSTM(
    256,
    return_sequences=True
)(caption_embedding)

# Generate next word
output = Dense(
    5000,
    activation='softmax'
)(lstm_output)

# Create caption model
caption_model = Model(
    [feature_input, caption_input],
    output
)

caption_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

caption_model.summary()
