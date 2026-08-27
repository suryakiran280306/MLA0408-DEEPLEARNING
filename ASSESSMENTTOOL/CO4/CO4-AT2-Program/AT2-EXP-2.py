import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.models import Model

# Input size
input_shape = (224, 224, 3)

# DenseNet backbone
base_model = DenseNet121(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape
)

# Feature extraction
x = base_model.output

# PixelNet-style decoder
x = UpSampling2D((2, 2))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)

x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)

# Output classes
output = Conv2D(
    5,
    (1, 1),
    activation='softmax',
    padding='same'
)(x)

# Create model
model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
