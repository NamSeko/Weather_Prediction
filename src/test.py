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
from src.split_data import create_inout_sequences, create_inout_sequences_2feature

warnings.filterwarnings("ignore")

device = setting.device
print(f"Using device: {device}")

test_data = pd.read_csv(setting.test_path)
scaler = setting.scaler

test = test_data.copy()
# test['time'] = pd.to_datetime(test['time'], format='%Y-%m-%d')
test['time'] = pd.to_datetime(test['time'], format='ISO8601')

test = test.set_index('time')
test = test.sort_index(ascending=True)

# Define hyperparameters
batch_size = setting.batch_size  # Number of samples per batch

seq_length = setting.seq_length  # Length of the input sequence

# model = setting.LSTM_model.to(device)
model = setting.Transformer_model.to(device)

state_dict = torch.load('./src/model/'+setting.path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Create input sequences for the test data
# X_test, y_test = create_inout_sequences(test_data, seq_length)
X_test, y_test = create_inout_sequences_2feature(test_data, seq_length)

# Create DataLoader for the test data
test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Evaluate the model on the test data
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())

predictions = np.concatenate(predictions, axis=0)
# print(predictions.shape)
# extra_feature = test.values[:predictions.shape[0], 1:]
# predictions = np.concatenate((predictions, extra_feature), axis=1)
# print(predictions.shape)

predictions = scaler.inverse_transform(predictions)
# predictions = predictions[:, 0]

# y_test = y_test.cpu().numpy()
# y_test = y_test.reshape(-1, 1)
# y_test = np.concatenate((y_test, extra_feature), axis=1)
y_test = scaler.inverse_transform(test[setting.seq_length:])
# y_test = y_test[:, 0]

data_df = pd.DataFrame(data={'Prediction Temp': predictions[:, 0], 'Actual Temp': y_test[:, 0], 'Prediction Rh': predictions[:, 1], 'Actual Rh': y_test[:, 1]}, index=test.index[-predictions.shape[0]:])
# Plot the predictions and actual values
def plot_temp(data_df):
    plt.figure(figsize=(12, 10))
    plt.plot(data_df['Prediction Temp'], label='Predicted', color='red')
    plt.plot(data_df['Actual Temp'], label='Actual', color='blue')
    plt.title('Weather Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Temperature')
    plt.legend()
    # plt.savefig(setting.images_path + 'weather_lstm_temp.png')
    plt.savefig(setting.images_path + 'weather_transformer_temp.png')
    plt.show()
    
def plot_rh(data_df):
    plt.figure(figsize=(12, 10))
    plt.plot(data_df['Prediction Rh'], label='Predicted', color='red')
    plt.plot(data_df['Actual Rh'], label='Actual', color='blue')
    plt.title('Weather Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Relative Humidity')
    plt.legend()
    # plt.savefig(setting.images_path + 'weather_lstm_rh.png')
    plt.savefig(setting.images_path + 'weather_transformer_rh.png')
    plt.show()

def evaluate_model(y_true, y_pred, name=""):
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

# evaluate_model(y_test, predictions, name="LSTM")
evaluate_model(y_test, predictions, name="Transformer")
plot_temp(data_df)
plot_rh(data_df)