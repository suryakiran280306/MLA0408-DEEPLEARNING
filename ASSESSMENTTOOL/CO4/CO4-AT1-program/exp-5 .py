import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    RepeatVector,
    TimeDistributed
)
from tensorflow.keras.models import Model

# -----------------------------
# CNN FEATURE EXTRACTOR
# -----------------------------

cnn = MobileNetV2(
    weights="imagenet",
    include_top=False,
    pooling="avg"
)

# Freeze CNN layers
cnn.trainable = False


# -----------------------------
# INPUTS
# -----------------------------

# Video feature sequence
video_input = Input(
    shape=(20, 1280)
)

# Caption input
caption_input = Input(
    shape=(15,)
)


# -----------------------------
# VIDEO ENCODER
# -----------------------------

video_encoder = LSTM(
    512,
    return_sequences=False
)(video_input)


# -----------------------------
# CAPTION DECODER
# -----------------------------

x = RepeatVector(15)(video_encoder)

x = LSTM(
    512,
    return_sequences=True
)(x)


# -----------------------------
# OUTPUT
# -----------------------------

VOCAB_SIZE = 5000

output = TimeDistributed(
    Dense(
        VOCAB_SIZE,
        activation="softmax"
    )
)(x)


# Create model
model = Model(
    inputs=[video_input, caption_input],
    outputs=output
)

# Compile
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
