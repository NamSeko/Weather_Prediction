import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib

# Paths to train val, test hourly data
scaler_hourly = joblib.load('./data/scaler_hourly.save')
scaler_daily = joblib.load('./data/scaler_daily.save')

train_hourly_path = './data/hourly/train_data.csv'
val_hourly_path = './data/hourly/val_data.csv'
test_hourly_path = './data/hourly/test_data.csv'

train_daily_path = './data/daily/train_data.csv'
val_daily_path = './data/daily/val_data.csv'
test_daily_path = './data/daily/test_data.csv'


# Paths to save the daily images
images_daily_path = './src/image/daily/'

# Paths to save the hourly images
images_hourly_path = './src/image/hourly/'

train_size = 0.8
val_size = 0.1
test_size = 0.1

# Ví dụ param grid cho LSTM
param_lstm = {
    'input_size': [11, 20],  # [hourly_features, daily_features]
    'hidden_size': 64,
    'output_size': [2, 4],  # [hourly_outputs, daily_outputs]
    'num_layers': 3,
    'dropout': 0.2,
    'learning_rate': 0.0001,
}

param_transformer = {
    'input_size': [11, 20],  # [hourly_features, daily_features]
    'd_model': 128,
    'num_head': 4,
    'num_layers_transformer': 4,
    'output_size': [2, 4],  # [hourly_outputs, daily_outputs]
    'dropout': 0.1,
    'learning_rate': 0.0005,
}