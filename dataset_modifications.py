
import pandas as pd
import numpy as np


def get_dataset(x):

    dataset = pd.read_csv(x)

    # Convert the 'start' column to datetime
    dataset['start'] = pd.to_datetime(dataset['start'])

    # Set the 'start' column as the index of the DataFrame
    dataset.set_index('start', inplace=True)

    # Resample the DataFrame to hourly frequency and calculate the mean load for each hour
    dataset = dataset.resample('h').mean()

    # Drop the last row of the DataFrame
    dataset.drop(dataset.tail(1).index, inplace=True)

    return dataset


def last_one_year_data(df):
    # Convert the index to datetime if it's not already
    df.index = pd.to_datetime(df.index)
    
    # Get the last date in the DataFrame
    last_date = df.index[-1]
    
    # Calculate the date three months ago
    three_months_ago = last_date - pd.DateOffset(years=2)
    
    # Filter the DataFrame to include only rows from the last three months
    last_three_months_data = df[df.index >= three_months_ago]
    
    return last_three_months_data

def last_three_months_data(df):
    # Convert the index to datetime if it's not already
    df.index = pd.to_datetime(df.index)
    
    # Get the last date in the DataFrame
    last_date = df.index[-1]
    
    # Calculate the date three months ago
    three_months_ago = last_date - pd.DateOffset(months=3)
    
    # Filter the DataFrame to include only rows from the last three months
    last_three_months_data = df[df.index >= three_months_ago]
    
    return last_three_months_data

def get_last_two_months_data(df):
    # Convert the index to datetime if it's not already
    df.index = pd.to_datetime(df.index)
    
    # Get the last date in the DataFrame
    last_date = df.index[-1]
    
    # Calculate the date three months ago
    three_months_ago = last_date - pd.DateOffset(months=2)
    
    # Filter the DataFrame to include only rows from the last three months
    last_three_months_data = df[df.index >= three_months_ago]
    
    return get_last_two_months_data

def get_last_month_data(df):
    # Convert the index to datetime if it's not already
    df.index = pd.to_datetime(df.index)
    
    # Extract the year and month from the index
    last_month_year = df.index.year[-1]
    last_month_month = df.index.month[-1]
    
    # Filter the DataFrame to include only rows from the last month
    last_month_data = df[(df.index.year == last_month_year) & (df.index.month == last_month_month)]
    
    return last_month_data

def get_last_two_weeks_data(df):
    # Convert the index to datetime if it's not already
    df.index = pd.to_datetime(df.index)
    
    # Get the last date in the DataFrame
    last_date = df.index[-1]
    
    # Calculate the date two weeks ago
    two_weeks_ago = last_date - pd.DateOffset(weeks=2)
    
    # Filter the DataFrame to include only rows from the last two weeks
    last_two_weeks_data = df[df.index >= two_weeks_ago]
    
    return last_two_weeks_data


# Preparing the dataset for ML
def df_to_X_y(df, window_size):
    df_as_np = df.to_numpy()
    X = []
    y=[]
    for i in range(len(df_as_np)-window_size):
        row = df_as_np[i:i + window_size]
        X.append (row)
        label = df_as_np[i+5]
        y.append(label)
    return np.array(X), np.array(y)

def mldatasets(X,y):
    train_size = int(len(X)*0.8)
    val_size= int(len(X)*0.1)+train_size
    # Preparing the training, validation and test datasets
    
    X_train, y_train = X[:train_size], y[:train_size]
    
    X_val, y_val= X[train_size:val_size], y[train_size:val_size]
    
    X_test, y_test = X[val_size:], y[val_size:]

    return X_train, y_train, X_val, y_val, X_test, y_test 

def df_to_X_y_horizon(df, window_size, forecast_horizon):
    X, y = [], []
    for i in range(len(df) - window_size - forecast_horizon + 1):
        X.append(df.iloc[i:i+window_size].values)
        y.append(df.iloc[i+window_size:i+window_size+forecast_horizon].values)
    return np.array(X), np.array(y)