import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import seaborn as sns
from datetime import datetime

from src import setting
from src.train import create_inout_sequences_hourly, create_inout_sequences_daily
from src.model.WeatherLSTM import WeatherLSTM
from src.model.WeatherTransformer import WeatherTransformer
from src.predict_utils import add_features

warnings.filterwarnings("ignore")

def plot_predictions(data_df, image_path, is_daily=True):
    """
    Plot predictions vs actual values with improved visualization
    """
    plt.style.use('default')  # Use default matplotlib style
    fig, axes = plt.subplots(2, 2 if is_daily else 1, figsize=(15, 10))
    fig.suptitle('Weather Prediction Results', fontsize=16)
    
    if is_daily:
        # Temperature plot
        ax = axes[0, 0]
        ax.plot(data_df.index, data_df['Prediction temperature_2m_mean (°C)'], label='Predicted', color='red')
        ax.plot(data_df.index, data_df['Actual temperature_2m_mean (°C)'], label='Actual', color='blue')
        ax.set_title('Temperature Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True)
        
        # Humidity plot
        ax = axes[0, 1]
        ax.plot(data_df.index, data_df['Prediction relative_humidity_2m_mean (%)'], label='Predicted', color='red')
        ax.plot(data_df.index, data_df['Actual relative_humidity_2m_mean (%)'], label='Actual', color='blue')
        ax.set_title('Humidity Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Relative Humidity (%)')
        ax.legend()
        ax.grid(True)
        
        # Min Temperature plot
        ax = axes[1, 0]
        ax.plot(data_df.index, data_df['Prediction temperature_2m_min (°C)'], label='Predicted', color='red')
        ax.plot(data_df.index, data_df['Actual temperature_2m_min (°C)'], label='Actual', color='blue')
        ax.set_title('Minimum Temperature Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True)
        
        # Max Temperature plot
        ax = axes[1, 1]
        ax.plot(data_df.index, data_df['Prediction temperature_2m_max (°C)'], label='Predicted', color='red')
        ax.plot(data_df.index, data_df['Actual temperature_2m_max (°C)'], label='Actual', color='blue')
        ax.set_title('Maximum Temperature Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True)
    else:
        # Temperature plot for hourly data
        ax = axes[0]
        ax.plot(data_df.index, data_df['Prediction temperature_2m (°C)'], label='Predicted', color='red')
        ax.plot(data_df.index, data_df['Actual temperature_2m (°C)'], label='Actual', color='blue')
        ax.set_title('Temperature Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature (°C)')
        ax.legend()
        ax.grid(True)
        
        # Humidity plot for hourly data
        ax = axes[1]
        ax.plot(data_df.index, data_df['Prediction relative_humidity_2m (%)'], label='Predicted', color='red')
        ax.plot(data_df.index, data_df['Actual relative_humidity_2m (%)'], label='Actual', color='blue')
        ax.set_title('Humidity Prediction')
        ax.set_xlabel('Time')
        ax.set_ylabel('Relative Humidity (%)')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(image_path)
    plt.close()

def evaluate_model(y_true, y_pred, name, is_daily=True):
    """
    Đánh giá mô hình với các metric phổ biến trong dự báo thời tiết
    """
    metrics = {}
    
    # Temperature metrics
    y_temp_true = y_true[:, 0]
    y_temp_pred = y_pred[:, 0]
    metrics['temp'] = {
        'MAE': mean_absolute_error(y_temp_true, y_temp_pred),
        'RMSE': np.sqrt(mean_squared_error(y_temp_true, y_temp_pred)),
        'R2': r2_score(y_temp_true, y_temp_pred)
    }
    
    # Humidity metrics
    y_rh_true = y_true[:, 1]
    y_rh_pred = y_pred[:, 1]
    metrics['humidity'] = {
        'MAE': mean_absolute_error(y_rh_true, y_rh_pred),
        'RMSE': np.sqrt(mean_squared_error(y_rh_true, y_rh_pred)),
        'R2': r2_score(y_rh_true, y_rh_pred)
    }
    
    if is_daily:
        # Min Temperature metrics
        y_min_temp_true = y_true[:, 2]
        y_min_temp_pred = y_pred[:, 2]
        metrics['min_temp'] = {
            'MAE': mean_absolute_error(y_min_temp_true, y_min_temp_pred),
            'RMSE': np.sqrt(mean_squared_error(y_min_temp_true, y_min_temp_pred)),
            'R2': r2_score(y_min_temp_true, y_min_temp_pred)
        }
        
        # Max Temperature metrics
        y_max_temp_true = y_true[:, 3]
        y_max_temp_pred = y_pred[:, 3]
        metrics['max_temp'] = {
            'MAE': mean_absolute_error(y_max_temp_true, y_max_temp_pred),
            'RMSE': np.sqrt(mean_squared_error(y_max_temp_true, y_max_temp_pred)),
            'R2': r2_score(y_max_temp_true, y_max_temp_pred)
        }
    
    # Print results
    print(f"\nKết quả đánh giá cho mô hình {name}:")
    print("\nTemperature:")
    print(f"MAE: {metrics['temp']['MAE']:.2f} °C")
    print(f"RMSE: {metrics['temp']['RMSE']:.2f} °C")
    print(f"R²: {metrics['temp']['R2']:.2f}")
    
    print("\nHumidity:")
    print(f"MAE: {metrics['humidity']['MAE']:.2f} %")
    print(f"RMSE: {metrics['humidity']['RMSE']:.2f} %")
    print(f"R²: {metrics['humidity']['R2']:.2f}")
    
    if is_daily:
        print("\nMin Temperature:")
        print(f"MAE: {metrics['min_temp']['MAE']:.2f} °C")
        print(f"RMSE: {metrics['min_temp']['RMSE']:.2f} °C")
        print(f"R²: {metrics['min_temp']['R2']:.2f}")
        
        print("\nMax Temperature:")
        print(f"MAE: {metrics['max_temp']['MAE']:.2f} °C")
        print(f"RMSE: {metrics['max_temp']['RMSE']:.2f} °C")
        print(f"R²: {metrics['max_temp']['R2']:.2f}")
    
    return metrics

def predict_daily(model, path_model, batch_size, seq_length, device):
    test_data = pd.read_csv(setting.test_daily_path)
    test_data.dropna(inplace=True)
    scaler = setting.scaler_daily
    
    # Add features
    test_data = add_features(test_data, is_daily=True)
    
    test = test_data.copy()
    test['time'] = pd.to_datetime(test['time'], format='%Y-%m-%d')
    test = test.set_index('time')
    test = test.sort_index(ascending=True)
    
    state_dict = torch.load(path_model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    X_test, y_test = create_inout_sequences_daily(test_data, seq_length)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Only scale target features
    target_columns = ['temperature_2m_mean (°C)', 'relative_humidity_2m_mean (%)', 
                     'temperature_2m_max (°C)', 'temperature_2m_min (°C)']
    predictions = scaler.inverse_transform(predictions)
    y_target = scaler.inverse_transform(test[seq_length:][target_columns])
    
    df = pd.DataFrame(data={
        'Prediction Temp': predictions[:, 0], 
        'Actual Temp': y_target[:, 0], 
        'Prediction Rh': predictions[:, 1], 
        'Actual Rh': y_target[:, 1],
        'Prediction Min Temp': predictions[:, 2],
        'Actual Min Temp': y_target[:, 2],
        'Prediction Max Temp': predictions[:, 3],
        'Actual Max Temp': y_target[:, 3]
    }, index=test.index[-predictions.shape[0]:])
    
    # Rename columns to match actual feature names
    df = df.rename(columns={
        'Prediction Temp': 'Prediction temperature_2m_mean (°C)',
        'Actual Temp': 'Actual temperature_2m_mean (°C)',
        'Prediction Rh': 'Prediction relative_humidity_2m_mean (%)',
        'Actual Rh': 'Actual relative_humidity_2m_mean (%)',
        'Prediction Min Temp': 'Prediction temperature_2m_min (°C)',
        'Actual Min Temp': 'Actual temperature_2m_min (°C)',
        'Prediction Max Temp': 'Prediction temperature_2m_max (°C)',
        'Actual Max Temp': 'Actual temperature_2m_max (°C)'
    })
    
    image_path = setting.images_daily_path
    name = 'LSTM' if 'lstm' in path_model else 'Transformer'
    plot_predictions(df, image_path + f'daily_{name}_predictions.png', is_daily=True)
    evaluate_model(y_target, predictions, name, is_daily=True)

def predict_hourly(model, path_model, batch_size, seq_length, device):
    test_data = pd.read_csv(setting.test_hourly_path)
    scaler = setting.scaler_hourly
    
    # Add features
    test_data = add_features(test_data, is_daily=False)
    
    test = test_data.copy()
    test['time'] = pd.to_datetime(test['time'], format='ISO8601')
    test = test.set_index('time')
    test = test.sort_index(ascending=True)
    
    state_dict = torch.load(path_model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    
    X_test, y_test = create_inout_sequences_hourly(test_data, seq_length)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    predictions = []
    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    
    # Only scale target features
    target_columns = ['temperature_2m (°C)', 'relative_humidity_2m (%)']
    predictions = scaler.inverse_transform(predictions)
    y_target = scaler.inverse_transform(test[seq_length:][target_columns])
    
    df = pd.DataFrame(data={
        'Prediction Temp': predictions[:, 0], 
        'Actual Temp': y_target[:, 0], 
        'Prediction Rh': predictions[:, 1], 
        'Actual Rh': y_target[:, 1]
    }, index=test.index[-predictions.shape[0]:])
    
    # Rename columns to match actual feature names
    df = df.rename(columns={
        'Prediction Temp': 'Prediction temperature_2m (°C)',
        'Actual Temp': 'Actual temperature_2m (°C)',
        'Prediction Rh': 'Prediction relative_humidity_2m (%)',
        'Actual Rh': 'Actual relative_humidity_2m (%)'
    })
    
    image_path = setting.images_hourly_path
    name = 'LSTM' if 'lstm' in path_model else 'Transformer'
    plot_predictions(df, image_path + f'hourly_{name}_predictions.png', is_daily=False)
    evaluate_model(y_target, predictions, name, is_daily=False)

if __name__ == "__main__":
    # Hyperparameters
    batch_size = 32
    
    param_lstm = setting.param_lstm
    param_transformer = setting.param_transformer
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Predict daily data
    seq_length = 30  # Updated sequence length
    print("\nPredicting daily data with LSTM model...")
    model = WeatherLSTM(input_size=param_lstm['input_size'][1],
                        hidden_size=param_lstm['hidden_size'], 
                        output_size=param_lstm['output_size'][1], 
                        num_layers=param_lstm['num_layers'], 
                        dropout=param_lstm['dropout']).to(device)
    predict_daily(model, './src/model/daily/lstm.pth', batch_size, seq_length, device)
    
    print("\nPredicting daily data with Transformer model...")
    model = WeatherTransformer(input_size=param_transformer['input_size'][1], 
                               d_model=param_transformer['d_model'], 
                               nhead=param_transformer['num_head'], 
                               num_layers=param_transformer['num_layers_transformer'], 
                               output_size=param_transformer['output_size'][1], 
                               dropout=param_transformer['dropout']).to(device)    
    predict_daily(model, './src/model/daily/transformer.pth', batch_size, seq_length, device)
    
    # Predict hourly data
    seq_length = 120  # Updated sequence length (7 days)
    print("\nPredicting hourly data with LSTM model...")
    model = WeatherLSTM(input_size=param_lstm['input_size'][0],
                        hidden_size=param_lstm['hidden_size'], 
                        output_size=param_lstm['output_size'][0], 
                        num_layers=param_lstm['num_layers'], 
                        dropout=param_lstm['dropout']).to(device)
    predict_hourly(model, './src/model/hourly/lstm.pth', batch_size, seq_length, device)
    
    print("\nPredicting hourly data with Transformer model...")
    model = WeatherTransformer(input_size=param_transformer['input_size'][0], 
                               d_model=param_transformer['d_model'], 
                               nhead=param_transformer['num_head'], 
                               num_layers=param_transformer['num_layers_transformer'], 
                               output_size=param_transformer['output_size'][0], 
                               dropout=param_transformer['dropout']).to(device) 
    predict_hourly(model, './src/model/hourly/transformer.pth', batch_size, seq_length, device)
    