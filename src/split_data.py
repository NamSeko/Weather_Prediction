import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import warnings
import torch

train_size = 0.8
val_size = 0.1
test_size = 0.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

warnings.filterwarnings("ignore")

scaler = MinMaxScaler(feature_range=(-1, 1))

def create_inout_sequences_hourly(data, seq_length):
    data['time'] = pd.to_datetime(data['time'], format='ISO8601')
    columns = data.columns[1:]
    data = data[columns].copy()
    data = data.values
    X, y = [], []
    L = len(data)
    for i in range(seq_length, L):
        train_seq = data[i-seq_length:i]
        train_label = data[i]  # Assuming the label is the first feature
        X.append(train_seq)
        y.append(train_label)
    X, y = np.array(X), np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    return X_tensor, y_tensor

def create_inout_sequences_daily(data, seq_length):
    data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d')
    columns = data.columns[1:]
    data = data[columns].copy()
    data = data.values
    X, y = [], []
    L = len(data)
    for i in range(seq_length, L):
        train_seq = data[i-seq_length:i]
        train_label = data[i]  # Assuming the label is the first feature
        X.append(train_seq)
        y.append(train_label)
    X, y = np.array(X), np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    return X_tensor, y_tensor

def split_data(df, train_size=train_size, val_size=val_size, test_size=test_size, path='data/'):
    """
    Splits the DataFrame into train, validation, and test sets.

    Args:
        df (pd.DataFrame): The DataFrame to split.
        train_size (float): The proportion of the data to include in the train set.
        val_size (float): The proportion of the data to include in the validation set.
        test_size (float): The proportion of the data to include in the test set.
        path (str): The path to save the split data.

    Returns:
        tuple: A tuple containing the train, validation, and test DataFrames.
    """

    # Calculate the sizes of each split
    total_size = len(df)
    train_end = int(train_size * total_size)
    val_end = int((train_size + val_size) * total_size)

    # Split the DataFrame
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    train_df = scaler.fit_transform(train_df)
    val_df = scaler.transform(val_df)
    test_df = scaler.transform(test_df)
        
    if not os.path.exists(path):
        os.makedirs(path)
    
    train_df = pd.DataFrame(train_df, columns=df.columns, index=df.index[:train_end])
    val_df = pd.DataFrame(val_df, columns=df.columns, index=df.index[train_end:val_end])
    test_df = pd.DataFrame(test_df, columns=df.columns, index=df.index[val_end:])
    train_df.to_csv(path + 'train_data.csv', index=True)
    val_df.to_csv(path + 'val_data.csv', index=True)
    test_df.to_csv(path + 'test_data.csv', index=True)

if __name__ == "__main__":
    # Daily data
    df_daily = pd.read_csv('./data/daily_data.csv', skiprows=2)
    df_daily['time'] = pd.to_datetime(df_daily['time'], format='%Y-%m-%d')
    df_daily = df_daily.set_index('time')
    df_daily = df_daily.sort_index(ascending=True)
    
    split_data(df_daily, path='./data/daily/')
    joblib.dump(scaler, './data/scaler_daily.save')
    
    # Hourly data
    df_hourly = pd.read_csv('./data/hourly_data.csv', skiprows=2)
    df_hourly['time'] = pd.to_datetime(df_hourly['time'], format='ISO8601')
    df_hourly = df_hourly.set_index('time')
    df_hourly = df_hourly.sort_index(ascending=True)
    
    split_data(df_hourly, path='./data/hourly/')
    joblib.dump(scaler, './data/scaler_hourly.save')