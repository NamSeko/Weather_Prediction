import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

from src import setting
from src.split_data import create_inout_sequences_hourly, create_inout_sequences_daily
from src.model.WeatherLSTM import WeatherLSTM
from src.model.WeatherTransformer import WeatherTransformer

warnings.filterwarnings("ignore")

def plot_temp(data_df, image_path):
    plt.figure(figsize=(12, 10))
    plt.plot(data_df['Prediction Temp'], label='Predicted', color='red')
    plt.plot(data_df['Actual Temp'], label='Actual', color='blue')
    plt.title('Weather Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature')
    plt.legend()
    plt.savefig(image_path)
    
def plot_rh(data_df, image_path):
    plt.figure(figsize=(12, 10))
    plt.plot(data_df['Prediction Rh'], label='Predicted', color='red')
    plt.plot(data_df['Actual Rh'], label='Actual', color='blue')
    plt.title('Weather Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Relative Humidity')
    plt.legend()
    plt.savefig(image_path)
    
def plot_min_temp(data_df, image_path):
    plt.figure(figsize=(12, 10))
    plt.plot(data_df['Prediction Min Temp'], label='Predicted', color='red')
    plt.plot(data_df['Actual Min Temp'], label='Actual', color='blue')
    plt.title('Weather Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Min Temperature')
    plt.legend()
    plt.savefig(image_path)

def plot_max_temp(data_df, image_path):
    plt.figure(figsize=(12, 10))
    plt.plot(data_df['Prediction Max Temp'], label='Predicted', color='red')
    plt.plot(data_df['Actual Max Temp'], label='Actual', color='blue')
    plt.title('Weather Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Max Temperature')
    plt.legend()
    plt.savefig(image_path)
    
def evaluate_model(y_true, y_pred, name):
    """
    Đánh giá mô hình với các metric phổ biến trong dự báo thời tiết
    """
    y_temp_true = y_true[:, 0]
    y_temp_pred = y_pred[:, 0]
    y_rh_true = y_true[:, 1]
    y_rh_pred = y_pred[:, 1]
    mae_temp = mean_absolute_error(y_temp_true, y_temp_pred)
    rmse_temp = np.sqrt(mean_squared_error(y_temp_true, y_temp_pred))
    r2_temp = r2_score(y_temp_true, y_temp_pred)
    
    mae_rh = mean_absolute_error(y_rh_true, y_rh_pred)
    rmse_rh = np.sqrt(mean_squared_error(y_rh_true, y_rh_pred))
    r2_rh = r2_score(y_rh_true, y_rh_pred)
    

    print(f"Kết quả đánh giá cho mô hình: {name}")
    print(f"MAE:")
    print(f"    Temp: {mae_temp:.2f} °C")
    print(f"    RH  : {mae_rh:.2f} %")
    print(f"RMSE:")
    print(f"    Temp: {rmse_temp:.2f} °C")
    print(f"    RH  : {rmse_rh:.2f} %")
    print(f"R²:")
    print(f"    Temp: {r2_temp:.2f}")
    print(f"    RH  : {r2_rh:.2f}")
    
    # return {
    #     "MAE": mae,
    #     "RMSE": rmse,
    #     "R2": r2,
    #     "MAPE": mape_val,
    #     "SMAPE": smape_val
    # }
    
def predict_daily(model, path_model, batch_size, seq_length, device):
    test_data = pd.read_csv(setting.test_daily_path)
    test_data.dropna(inplace=True)
    scaler = setting.scaler_daily
    
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
    predictions = scaler.inverse_transform(predictions)
    y_target = scaler.inverse_transform(test[seq_length:])
    df = pd.DataFrame(data={'Prediction Temp': predictions[:, 0], 
                            'Actual Temp': y_target[:, 0], 
                            'Prediction Rh': predictions[:, 1], 
                            'Actual Rh': y_target[:, 1],
                            'Prediction Min Temp': predictions[:, 2],
                            'Actual Min Temp': y_target[:, 2],
                            'Prediction Max Temp': predictions[:, 3],
                            'Actual Max Temp': y_target[:, 3]},
                      index=test.index[-predictions.shape[0]:])
    image_path = setting.images_daily_path
    
    if 'lstm' in path_model:
        name = 'LSTM'
    else:
        name = 'Transformer'
    plot_temp(df, image_path + f'daily_{name}_temp.png')
    plot_rh(df, image_path + f'daily_{name}_rh.png')
    plot_min_temp(df, image_path + f'daily_{name}_min_temp.png')
    plot_max_temp(df, image_path + f'daily_{name}_max_temp.png')
    evaluate_model(y_target, predictions, name)

def predict_hourly(model, path_model, batch_size, seq_length, device):
    test_data = pd.read_csv(setting.test_hourly_path)
    scaler = setting.scaler_hourly
    
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
    predictions = scaler.inverse_transform(predictions)
    y_target = scaler.inverse_transform(test[seq_length:])
    df = pd.DataFrame(data={'Prediction Temp': predictions[:, 0], 
                            'Actual Temp': y_target[:, 0], 
                            'Prediction Rh': predictions[:, 1], 
                            'Actual Rh': y_target[:, 1]},
                      index=test.index[-predictions.shape[0]:])
    image_path = setting.images_hourly_path
    if 'lstm' in path_model:
        name = 'LSTM'
    else:
        name = 'Transformer'
    plot_temp(df, image_path + f'hourly_{name}_temp.png')
    plot_rh(df, image_path + f'hourl_{name}_rh.png')
    evaluate_model(y_target, predictions, name)

if __name__ == "__main__":
    # Hyperparameters
    learning_rate = 0.0001  # Learning rate for the optimizer
    batch_size = 64  # Number of samples per batch
    seq_length = 60  # Length of the input sequence
    num_epochs = 150  # Number of epochs to train the model
    hidden_size = 64  # Number of features in the hidden state
    num_layers = 2  # Number of recurrent layers
    d_model = 64  # Dimension of the model for Transformer
    num_heads = 2  # Number of attention heads for Transformer
    num_layers_transformer = 2  # Number of layers for Transformer
    dropout = 0.3  # Dropout rate for Transformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Predict daily data
    input_size, output_size = 4, 4
    # LSTM
    print("Predicting daily data with LSTM model...")
    model = WeatherLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout).to(device)
    predict_daily(model, './src/model/daily/lstm.pth', batch_size, seq_length, device)
    # Transformer
    print("Predicting daily data with Transformer model...")
    model = WeatherTransformer(input_size=input_size, d_model=d_model, nhead=num_heads, num_layers=num_layers_transformer, output_size=output_size, dropout=dropout).to(device)
    predict_daily(model, './src/model/daily/transformer.pth', batch_size, seq_length, device)
    # Predict hourly data
    input_size, output_size = 2, 2
    # LSTM
    print("Predicting hourly data with LSTM model...")
    model = WeatherLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout).to(device)
    predict_hourly(model, './src/model/hourly/lstm.pth', batch_size, seq_length, device)
    # Transformer
    print("Predicting hourly data with Transformer model...")
    model = WeatherTransformer(input_size=input_size, d_model=d_model, nhead=num_heads, num_layers=num_layers_transformer, output_size=output_size, dropout=dropout).to(device)
    predict_hourly(model, './src/model/hourly/transformer.pth', batch_size, seq_length, device)
    