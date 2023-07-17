# https://medium.com/@polanitzer/predicting-the-israeli-lottery-results-for-the-november-29-2022-game-using-an-artificial-191489eb2c10
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import date
import ephem
import json



# Define the pool of numbers and the sequence length
number_pool = range(1, 71)
sequence_length = 7

# Generate training data with lunar phase
def generate_data_with_lunar_phase():
    file="seventh_version.json"
    with open('number_data/'+file,'r') as f:
        data=json.loads(f.read())
        f.close()
    X = []
    y = []
    # for _ in range(1000):  # Generate 1000 sequences for training
    #     sequence = np.random.choice(number_pool, size=sequence_length, replace=False)
    #     target = np.random.choice(number_pool)
    #     X.append(np.append(sequence, target))
    #     y.append(np.random.choice(number_pool, size=sequence_length, replace=False))

    # Add lunar phase as the 8th input
    X = np.array(X)
    lunar_phases = []
    for _ in range(X.shape[0]):
        today = date.today()
        moon = ephem.Moon()
        moon.compute(today)
        lunar_phases.append(moon.phase / 100.0)  # Normalize lunar phase to range [0, 1]
    X_with_lunar_phase = np.column_stack((X, lunar_phases))
    return X_with_lunar_phase, np.array(y)

# Create the LSTM model with lunar phase
def create_lstm_model_with_lunar_phase():
    model = keras.Sequential()
    model.add(layers.LSTM(64, input_shape=(sequence_length + 2, 1)))  # "+2" for the known 8th input and the lunar phase
    model.add(layers.Dense(sequence_length, activation='linear'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Create the transformer model with lunar phase
def create_transformer_model_with_lunar_phase():
    inputs = layers.Input(shape=(sequence_length + 2,), dtype=tf.int32)  # "+2" for the known 8th input and the lunar phase
    embeddings = layers.Embedding(input_dim=71, output_dim=64)(inputs[:, :-1])  # Excluding the lunar phase from embeddings
    transformer_output = layers.Transformer(num_layers=2, d_model=64, num_heads=4,
                                            activation='relu')(embeddings)
    outputs = layers.Dense(sequence_length, activation='linear')(transformer_output)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Generate training data with lunar phase
X_train, y_train = generate_data_with_lunar_phase()

# Create and train the LSTM model with lunar phase
lstm_model = create_lstm_model_with_lunar_phase()
lstm_model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train,
               epochs=10, batch_size=32)

# Create and train the transformer model with lunar phase
transformer_model = create_transformer_model_with_lunar_phase()
transformer_model.fit(X_train, y_train, epochs=10, batch_size=32)

# Generate a new sequence with the known 8th input (lunar phase) using a combination of LSTM and transformer
def generate_sequence_with_lunar_phase(known_input):
    sequence = np.random.choice(number_pool, size=sequence_length, replace=False)
    input_sequence = np.append(sequence, known_input)

    # Calculate lunar phase for the current date
    today = date.today()
    moon = ephem.Moon()
    moon.compute(today)
    lunar_phase = moon.phase / 100.0  # Normalize lunar phase to range [0, 1]

    input_data = np.append(input_sequence, lunar_phase)
    input_data_lstm = input_data.reshape((1, sequence_length + 2, 1))
    input_data_transformer = np.expand_dims(input_data, axis=0)

    # Generate LSTM prediction
    lstm_prediction = lstm_model.predict(input_data_lstm)

    # Generate transformer prediction
    transformer_prediction = transformer_model.predict(input_data_transformer)

    # Combine LSTM and transformer predictions
    combined_prediction = (lstm_prediction + transformer_prediction) / 2.0

    return combined_prediction.flatten()

# Example usage
known_input = 42
predicted_sequence = generate_sequence_with_lunar_phase(known_input)
print(f"Known Input: {known_input}")
print(f"Predicted Sequence: {predicted_sequence}")
