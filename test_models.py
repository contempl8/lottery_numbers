import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import json
import dateparser
import ephem
import pandas as pd

# DataFrame Header
df_header = {
    'Date': [],
    'b1': [],
    'b2': [],
    'b3': [],
    'b4': [],
    'b5': [],
    'p1': [],
    'weekday': [],
    'dofw': [],
    'month': [],
    'moon': [],
}
# Define the input and output mappings
input_mapping = [
    range(1, 8),    # Index 0: Range 1 to 7
    range(1, 366),  # Index 1: Range 1 to 365
    range(1, 13),   # Index 2: Range 1 to 12
    range(101)      # Index 3: Range 0 to 100
]
output_mapping = [
    range(1, 71),   # First 5 numbers: Range 1 to 70 (no repetition)
    range(1, 13)    # 6th number: Range 1 to 12
]

def format_input_date(date):
    d=dateparser.parse(date)
    # Get the number of the day of the week (Monday is 0 and Sunday is 6)
    weekday=d.weekday()
    # Get the day of the year (ranging from 1 to 365 or 366 in a leap year)
    day_of_week=d.timetuple().tm_yday
    # Get the month number (ranging from 1 to 12)
    month_number=d.month
    moon = ephem.Moon()
    moon.compute(d)
    return [weekday/6.0, day_of_week/366.0, month_number/12.0,moon.phase/100.0]
# Generate training data
def generate_data():
    X = []
    y = []
    file="seventh_version.json"
    with open('number_data/'+file,'r') as f:
        data=json.loads(f.read())
        f.close()
    for d_name,b in data.items():
        b[0].append(b[1])
        to_floats = [float(x) for x in b[0]]
        y.append(to_floats)
        df_header["Date"].append(d_name)
        df_header["b1"].append(b[0][0])
        df_header["b2"].append(b[0][1])
        df_header["b3"].append(b[0][2])
        df_header["b4"].append(b[0][3])
        df_header["b5"].append(b[0][4])
        df_header["p1"].append(b[0][5])
        ret=format_input_date(d_name)
        df_header["weekday"].append(ret[0])
        df_header["dofw"].append(ret[1])
        df_header["month"].append(ret[2])
        df_header["moon"].append(ret[3])
        # X.append(format_input_date(d_name))
    df = pd.DataFrame(df_header)
    df1 = df.copy(deep=True)
    df1.drop(['Date','dofw','month','moon'], axis=1, inplace=True)
    number_of_features = df1.shape[1]
    window_length = 7
    train_rows = df1.values.shape[0]
    train_samples = np.empty([ train_rows - window_length, window_length, number_of_features], dtype=float)
    train_labels = np.empty([ train_rows - window_length, number_of_features], dtype=float)
    for i in range(0, train_rows-window_length):
        train_samples[i] = df1.iloc[i : i+window_length, 0 : number_of_features]
        train_labels[i] = df1.iloc[i+window_length : i+window_length+1, 0 : number_of_features]
    return np.array(X), np.array(y)

# Create the LSTM model
def create_lstm_model():
    model = keras.Sequential()
    model.add(layers.LSTM(64, input_shape=(4, 1)))  # 4 input indices, input shape is adjusted for LSTM
    model.add(layers.Dense(6, activation='linear'))  # 6 output numbers
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Custom Transformer Encoder Layer
class TransformerEncoderLayer(keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.mha = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = keras.Sequential([
            keras.layers.Dense(dff, activation='relu'),
            keras.layers.Dense(d_model)
        ])

        self.layer_norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layer_norm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2

# Create the transformer model
def create_transformer_model():
    inputs = keras.Input(shape=(4,), dtype=tf.int32)  # 4 input indices
    embeddings = keras.layers.Embedding(input_dim=366, output_dim=64)(inputs)  # Embedding size is chosen as 366 for flexibility
    encoder_layer = TransformerEncoderLayer(d_model=64, num_heads=4, dff=128)
    encoder_output = encoder_layer(embeddings)
    outputs = keras.layers.Dense(6, activation='linear')(encoder_output)  # 6 output numbers
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

# Generate training data
X_train, y_train = generate_data()

# Create and train the LSTM model
lstm_model = create_lstm_model()
lstm_model.fit(X_train.reshape((X_train.shape[0], X_train.shape[1], 1)), y_train,
               epochs=150, batch_size=32, verbose=1)

# Create and train the transformer model
# transformer_model = create_transformer_model()
# transformer_model.fit(X_train, y_train, epochs=10, batch_size=16)

# Generate a new sequence using both LSTM and transformer models
def generate_sequence(date):
    # sequence = [np.random.choice(mapping) for mapping in input_mapping]
    # input_data = np.array([sequence])
    input_data = np.array(format_input_date(date)).reshape((1, 4, 1))
    # Generate LSTM prediction
    lstm_prediction = lstm_model.predict(input_data)
    return lstm_prediction[0]
    # # Generate transformer prediction
    # transformer_prediction = transformer_model.predict(input_data)

    # # Combine LSTM and transformer predictions
    # combined_prediction = (lstm_prediction + transformer_prediction) / 2.0

    # return combined_prediction[0]

# Example usage
date_to_predict="Mon, Jun 12, 2023"
predicted_sequence = generate_sequence(date_to_predict)
print(f"Predicted Sequence: {predicted_sequence}")
