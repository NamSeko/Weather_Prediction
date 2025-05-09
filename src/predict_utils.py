import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta

def prepare_prediction_data(input_data, seq_length, is_daily=True):
    """
    Prepare input data for prediction
    Args:
        input_data: DataFrame containing the input data
        seq_length: Length of the input sequence
        is_daily: Boolean indicating if the data is daily or hourly
    Returns:
        Processed input tensor for prediction
    """
    # Add features to input data
    processed_data = add_features(input_data, is_daily)
    
    if is_daily:
        # Select features for daily prediction
        feature_columns = ['temperature_2m_mean (°C)', 'relative_humidity_2m_mean (%)', 
                          'temperature_2m_max (°C)', 'temperature_2m_min (°C)',
                          'month_sin', 'month_cos', 'day_sin', 'day_cos',
                          'temp_range', 'temperature_2m_mean (°C)_rolling_mean', 
                          'relative_humidity_2m_mean (%)_rolling_mean',
                          'temperature_2m_max (°C)_rolling_mean', 
                          'temperature_2m_min (°C)_rolling_mean',
                          'temperature_2m_mean (°C)_rolling_std', 
                          'relative_humidity_2m_mean (%)_rolling_std',
                          'temperature_2m_mean (°C)_lag_1', 
                          'relative_humidity_2m_mean (%)_lag_1',
                          'temperature_2m_mean (°C)_lag_7', 
                          'relative_humidity_2m_mean (%)_lag_7',
                          'temp_humidity_interaction']
    else:
        # Select features for hourly prediction
        feature_columns = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 
                          'hour_sin', 'hour_cos', 
                          'temperature_2m (°C)_rolling_mean', 
                          'relative_humidity_2m (%)_rolling_mean',
                          'temperature_2m (°C)_rolling_std', 
                          'relative_humidity_2m (%)_rolling_std',
                          'temperature_2m (°C)_lag_1', 
                          'relative_humidity_2m (%)_lag_1',
                          'temp_humidity_interaction']
    
    # Get the last seq_length rows for prediction
    X_data = processed_data[feature_columns].values[-seq_length:]
    
    # Convert to tensor
    X_tensor = torch.tensor(X_data, dtype=torch.float32).unsqueeze(0)  # Add batch dimension
    
    return X_tensor

def generate_future_dates(last_date, num_steps, is_daily=True):
    """
    Generate future dates for prediction
    Args:
        last_date: Last date in the input data
        num_steps: Number of future steps to predict
        is_daily: Boolean indicating if the data is daily or hourly
    Returns:
        List of future dates
    """
    future_dates = []
    if is_daily:
        for i in range(1, num_steps + 1):
            future_dates.append(last_date + timedelta(days=i))
    else:
        for i in range(1, num_steps + 1):
            future_dates.append(last_date + timedelta(hours=i))
    return future_dates

def process_prediction_output(predictions, future_dates, is_daily=True):
    """
    Process model predictions into a readable format
    Args:
        predictions: Model predictions
        future_dates: List of future dates
        is_daily: Boolean indicating if the data is daily or hourly
    Returns:
        DataFrame with predictions
    """
    if is_daily:
        columns = ['temperature_2m_mean (°C)', 'relative_humidity_2m_mean (%)', 
                  'temperature_2m_max (°C)', 'temperature_2m_min (°C)']
    else:
        columns = ['temperature_2m (°C)', 'relative_humidity_2m (%)']
    
    predictions_df = pd.DataFrame(predictions, columns=columns)
    predictions_df['time'] = future_dates
    
    return predictions_df

def add_features(data, is_daily=True):
    """
    Add additional features to the dataset
    Args:
        data: DataFrame containing the original data
        is_daily: Boolean indicating if the data is daily or hourly
    Returns:
        DataFrame with additional features
    """
    # Convert time column to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(data['time']):
        if is_daily:
            data['time'] = pd.to_datetime(data['time'], format='%Y-%m-%d')
        else:
            data['time'] = pd.to_datetime(data['time'], format='ISO8601')
    
    # Add time-based features
    data['day_of_year'] = data['time'].dt.dayofyear
    data['month'] = data['time'].dt.month
    data['day_of_week'] = data['time'].dt.dayofweek
    
    # Add cyclic features for time
    data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
    data['month_cos'] = np.cos(2 * np.pi * data['month']/12)
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_year']/365)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_year']/365)
    
    if is_daily:
        # For daily data
        # Temperature range
        data['temp_range'] = data['temperature_2m_max (°C)'] - data['temperature_2m_min (°C)']
        
        # Rolling statistics (7-day window)
        for col in ['temperature_2m_mean (°C)', 'relative_humidity_2m_mean (%)', 
                   'temperature_2m_max (°C)', 'temperature_2m_min (°C)']:
            data[f'{col}_rolling_mean'] = data[col].rolling(window=7, min_periods=1).mean()
            data[f'{col}_rolling_std'] = data[col].rolling(window=7, min_periods=1).std()
            
        # Lag features (previous days)
        for col in ['temperature_2m_mean (°C)', 'relative_humidity_2m_mean (%)', 
                   'temperature_2m_max (°C)', 'temperature_2m_min (°C)']:
            data[f'{col}_lag_1'] = data[col].shift(1)
            data[f'{col}_lag_7'] = data[col].shift(7)
            
        # Temperature-humidity interaction
        data['temp_humidity_interaction'] = data['temperature_2m_mean (°C)'] * data['relative_humidity_2m_mean (%)']
        
    else:
        # For hourly data
        data['hour'] = data['time'].dt.hour
        
        # Add cyclic features for hour
        data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
        data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
        
        # Rolling statistics (24-hour window)
        for col in ['temperature_2m (°C)', 'relative_humidity_2m (%)']:
            data[f'{col}_rolling_mean'] = data[col].rolling(window=24, min_periods=1).mean()
            data[f'{col}_rolling_std'] = data[col].rolling(window=24, min_periods=1).std()
            
        # Lag features (previous hours)
        for col in ['temperature_2m (°C)', 'relative_humidity_2m (%)']:
            data[f'{col}_lag_1'] = data[col].shift(1)
            data[f'{col}_lag_24'] = data[col].shift(24)
            
        # Temperature-humidity interaction
        data['temp_humidity_interaction'] = data['temperature_2m (°C)'] * data['relative_humidity_2m (%)']
    
    # Fill NaN values with forward fill
    data = data.fillna(method='ffill')
    
    return data 