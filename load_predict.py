# Import the necessary libraries
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import StandardScaler
from utils import get_formatted_data

# Load the saved model
model = keras.models.load_model('test_model95.keras')

window_length=7

df1 = get_formatted_data()
scaler = StandardScaler()
scaler.fit_transform(df1.values)
next_set=df1.tail((window_length))
next_set=np.array(next_set)
x_next=scaler.transform(next_set)

next_Date='Sometime'

y_next_pred = model.predict(np.array([x_next]))
print("The predicted numbers for the lottery game which will take place on",next_Date, "are (with rounding up):", scaler.inverse_transform(y_next_pred).astype(int)[0]+1)
print("The predicted numbers for the lottery game which will take place on",next_Date, "are (without rounding up):", scaler.inverse_transform(y_next_pred).astype(int)[0])