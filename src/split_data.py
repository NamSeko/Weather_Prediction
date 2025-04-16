import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import os
import warnings

warnings.filterwarnings("ignore")

scaler = MinMaxScaler(feature_range=(-1, 1))
joblib.dump(scaler, '../data/scaler.save')

def scale_data(df):
    """
    Scales the specified columns of the DataFrame to the range (-1, 1).

    Args:
        df (pd.DataFrame): The DataFrame containing the data to scale.
        columns_to_scale (list): List of column names to scale.

    Returns:
        pd.DataFrame: A new DataFrame with the specified columns scaled.
    """
    # Scale the specified columns
    df_scaled = scaler.fit_transform(df)

    return df_scaled

def split_data(df, train_size=0.8, val_size=0.1, test_size=0.1, path='data/'):
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
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    # Save the splits to .npy files
    np.save(path + 'train_data.npy', train_df)
    np.save(path + 'val_data.npy', val_df)
    np.save(path + 'test_data.npy', test_df)

if __name__ == "__main__":
    # Daily data
    df_daily = pd.read_csv('../data/data_daily-2010_2025.csv', skiprows=2)
    df_daily['time'] = pd.to_datetime(df_daily['time'], format='%Y-%m-%d')
    df_daily = df_daily.set_index('time')
    df_daily = df_daily.sort_index(ascending=True)
    
    daily_data_temp = df_daily[['temperature_2m_mean (°C)', 'soil_temperature_0_to_7cm_mean (°C)', 'et0_fao_evapotranspiration_sum (mm)', 'relative_humidity_2m_mean (%)', 'soil_moisture_0_to_7cm_mean (m³/m³)']].copy()
    df_daily_temp_scaled = scale_data(daily_data_temp)
    split_data(df_daily_temp_scaled, path='../data/daily/temp/')
    
    daily_data_rh = df_daily[['relative_humidity_2m_mean (%)', 'soil_moisture_0_to_7cm_mean (m³/m³)', 'cloud_cover_mean (%)', 'temperature_2m_mean (°C)', 'soil_temperature_0_to_7cm_mean (°C)', 'et0_fao_evapotranspiration_sum (mm)']].copy()
    daily_data_rh_scaled = scale_data(daily_data_rh)
    split_data(daily_data_rh_scaled, path='../data/daily/rh/')
    
    # Hourly data
    df_hourly = pd.read_csv('../data/data_hourly-2023_2025.csv', skiprows=2)
    df_hourly['time'] = pd.to_datetime(df_hourly['time'], format='ISO8601')
    df_hourly = df_hourly.set_index('time')
    df_hourly = df_hourly.sort_index(ascending=True)
    
    hourly_data = df_hourly[['temperature_2m (°C)', 'soil_temperature_0_to_7cm (°C)', 'et0_fao_evapotranspiration (mm)', 'relative_humidity_2m (%)', 'soil_moisture_0_to_7cm (m³/m³)']].copy()
    df_hourly_scaled = scale_data(hourly_data)
    split_data(df_hourly_scaled, path='../data/hourly/')