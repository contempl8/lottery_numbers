# https://medium.com/@polanitzer/predicting-the-israeli-lottery-results-for-the-november-29-2022-game-using-an-artificial-191489eb2c10

import json
import dateparser
import ephem

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout, LayerNormalization, MultiHeadAttention, Conv1D, Input
from tensorflow.keras.optimizers import Adam, Nadam, Adadelta
from tensorflow.keras.metrics import mse

from utils import get_scaled_training_data_x_y, get_formatted_data

physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0], True)
window_length=32
x_train, y_train, window_length, number_of_features, scaler = get_scaled_training_data_x_y(window_length=window_length)

def transformer(s_model):
    # Begin Transformer
    x = MultiHeadAttention(key_dim=number_of_features, num_heads=4, dropout=0.01)(s_model,s_model)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Conv1D(filters=4, kernel_size=1, activation='relu')(x)
    x = Dropout(0.004)(x)
    x = Conv1D(filters=6, kernel_size=1)(x)
    return Dense(number_of_features)(x)

# Initialising the RNN
model = Sequential()
# Adding the input layer and the LSTM layer
model.add(Bidirectional(LSTM(240,
                        input_shape = (window_length, number_of_features),
                        return_sequences = True)))
# Adding a first Dropout layer
model.add(Dropout(0.08))
# Adding a second LSTM layer
model.add(Bidirectional(LSTM(240,
                        input_shape = (window_length, number_of_features),
                        return_sequences = True)))
# Adding a second Dropout layer
model.add(Dropout(0.04))
# Adding a third LSTM layer
model.add(Bidirectional(LSTM(240,
                        input_shape = (window_length, number_of_features),
                        return_sequences = True)))
# Adding a fourth LSTM layer
# Adding a third Dropout layer
model.add(Dropout(0.005))
model.add(Bidirectional(LSTM(240,
                        input_shape = (window_length, number_of_features),
                        return_sequences = True)))

# Adding the first output layer
model.add(Dense(25))
# Adding the last output layer
model.add(Dense(number_of_features))
inputs = Input(shape=(window_length,number_of_features))
x=model(inputs)
outputs=transformer(x)
model2=Model(inputs,outputs)

model2.compile(optimizer=Nadam(learning_rate=0.001), loss ='mse', metrics=['accuracy'])
model2.fit(x=x_train, y=y_train, batch_size=300, epochs=1200, verbose=2)
model2.save(f'test_model95_window{window_length}.keras')

df1 = get_formatted_data()

next_set=df1.tail((window_length))
next_set=np.array(next_set)
x_next=scaler.transform(next_set)

next_Date='Sometime'

y_next_pred = model.predict(np.array([x_next]))
print("The predicted numbers for the lottery game which will take place on",next_Date, "are (with rounding up):", scaler.inverse_transform(y_next_pred).astype(int)[0]+1)
print("The predicted numbers for the lottery game which will take place on",next_Date, "are (without rounding up):", scaler.inverse_transform(y_next_pred).astype(int)[0])