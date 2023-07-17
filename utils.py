import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import json
import dateparser
import ephem

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

def get_raw_data():
    """
    Retrieves data from a JSON file and processes it into a DataFrame.

    Returns:
        df_header (dict): A dictionary containing processed data, with keys representing column names and values representing data points.

    Raises:
        FileNotFoundError: If the specified JSON file is not found.
        JSONDecodeError: If the contents of the JSON file cannot be decoded.

    """
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
    return df_header

def get_formatted_data() -> pd.DataFrame:
    """
    Format the given data into a pandas DataFrame.

    Parameters:
        data (list or dict): The data to be formatted.

    Returns:
        pandas.DataFrame: The formatted DataFrame.
    """
    df = pd.DataFrame(get_raw_data())
    df.drop(['Date','weekday','dofw','month','moon'], axis=1, inplace=True)
    return df

def get_scaled_training_data_x_y(window_length:int =7):
    df1 =get_formatted_data()
    number_of_features = df1.shape[1]
    train_rows = df1.values.shape[0]
    train_samples = np.empty([ train_rows - window_length, window_length, number_of_features], dtype=float)
    train_labels = np.empty([ train_rows - window_length, number_of_features], dtype=float)
    for i in range(0, train_rows-window_length):
        train_samples[i] = df1.iloc[i : i+window_length, 0 : number_of_features]
        train_labels[i] = df1.iloc[i+window_length : i+window_length+1, 0 : number_of_features]

    scaler = StandardScaler()
    transformed_dataset = scaler.fit_transform(df1.values)
    scaled_train_samples = pd.DataFrame(data=transformed_dataset, index=df1.index)

    x_train = np.empty([ train_rows - window_length, window_length, number_of_features], dtype=float)
    y_train = np.empty([ train_rows - window_length, number_of_features], dtype=float)

    for i in range(0, train_rows-window_length):
        x_train[i] = scaled_train_samples.iloc[i : i+window_length, 0 : number_of_features]
        y_train[i] = scaled_train_samples.iloc[i+window_length : i+window_length+1, 0 : number_of_features]

    return x_train, y_train, window_length, number_of_features, scaler