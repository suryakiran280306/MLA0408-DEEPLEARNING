import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding
from tensorflow.keras.layers import Bidirectional, LSTM, Dense

# Parameters
vocab_size = 10000
embedding_dim = 128
hidden_units = 256

# Encoder input
encoder_input = Input(shape=(None,))

# Embedding
encoder_embedding = Embedding(
    vocab_size,
    embedding_dim
)(encoder_input)

# Bidirectional LSTM Encoder
encoder_output, forward_h, forward_c, backward_h, backward_c = Bidirectional(
    LSTM(hidden_units, return_state=True)
)(encoder_embedding)

# Combine forward and backward states
state_h = tf.keras.layers.Concatenate()(
    [forward_h, backward_h]
)

state_c = tf.keras.layers.Concatenate()(
    [forward_c, backward_c]
)

# Decoder input
decoder_input = Input(shape=(None,))

decoder_embedding = Embedding(
    vocab_size,
    embedding_dim
)(decoder_input)

# Decoder LSTM
decoder_lstm = LSTM(
    hidden_units * 2,
    return_sequences=True
)

decoder_output = decoder_lstm(
    decoder_embedding,
    initial_state=[state_h, state_c]
)

# Output layer
output = Dense(
    vocab_size,
    activation='softmax'
)(decoder_output)

# Create model
model = Model(
    [encoder_input, decoder_input],
    output
)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
