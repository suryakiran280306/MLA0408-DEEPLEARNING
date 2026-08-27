import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Model

# Number of segmentation classes
NUM_CLASSES = 8

# Load DenseNet without classification layer
base_model = DenseNet121(
    include_top=False,
    weights="imagenet",
    input_shape=(256, 256, 3)
)

# Freeze some layers
for layer in base_model.layers[:200]:
    layer.trainable = False

# DenseNet feature output
x = base_model.output

# Pixel-wise prediction / decoder
x = Conv2D(256, (3, 3), activation="relu", padding="same")(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation="relu", padding="same")(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation="relu", padding="same")(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation="relu", padding="same")(x)

# Final pixel-wise classification
output = Conv2D(
    NUM_CLASSES,
    (1, 1),
    activation="softmax",
    padding="same"
)(x)

# Create model
model = Model(
    inputs=base_model.input,
    outputs=output
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
