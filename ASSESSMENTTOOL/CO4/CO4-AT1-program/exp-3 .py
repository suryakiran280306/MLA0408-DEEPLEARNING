import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Embedding,
    Bidirectional,
    LSTM,
    Dense
)

# Vocabulary sizes
input_vocab_size = 10000
target_vocab_size = 10000

# Maximum sequence length
max_length = 20

# Encoder input
encoder_inputs = Input(shape=(max_length,))

# Embedding layer
encoder_embedding = Embedding(
    input_vocab_size,
    256
)(encoder_inputs)

# Bidirectional LSTM Encoder
encoder_output, forward_h, forward_c, backward_h, backward_c = \
    Bidirectional(
        LSTM(256, return_state=True)
    )(encoder_embedding)

# Combine forward and backward states
state_h = tf.keras.layers.Concatenate()(
    [forward_h, backward_h]
)

state_c = tf.keras.layers.Concatenate()(
    [forward_c, backward_c]
)

# Decoder input
decoder_inputs = Input(shape=(max_length,))

decoder_embedding = Embedding(
    target_vocab_size,
    256
)(decoder_inputs)

# Decoder LSTM
decoder_lstm = LSTM(
    512,
    return_sequences=True,
    return_state=True
)

decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=[state_h, state_c]
)

# Output layer
decoder_outputs = Dense(
    target_vocab_size,
    activation="softmax"
)(decoder_outputs)

# Create model
model = Model(
    [encoder_inputs, decoder_inputs],
    decoder_outputs
)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
