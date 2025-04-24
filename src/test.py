import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import setting
from split_data import create_inout_sequences
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

device = setting.device
print(f"Using device: {device}")

data_train = pd.read_csv(setting.train_path)
data_val = pd.read_csv(setting.val_path)
test_data = pd.read_csv(setting.test_path)
scaler = setting.scaler

train = data_train.copy()
val = data_val.copy()
test = test_data.copy()
train['time'] = pd.to_datetime(train['time'], format='%Y-%m-%d')
val['time'] = pd.to_datetime(val['time'], format='%Y-%m-%d')
test['time'] = pd.to_datetime(test['time'], format='%Y-%m-%d')
train = train.set_index('time')
val = val.set_index('time')
test = test.set_index('time')
train = train.sort_index(ascending=True)
val = val.sort_index(ascending=True)
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
X_test, y_test = create_inout_sequences(test_data, seq_length)

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
extra_feature = test.values[:predictions.shape[0], 1:]
predictions = np.concatenate((predictions, extra_feature), axis=1)
# print(predictions.shape)

predictions = scaler.inverse_transform(predictions)
predictions = predictions[:, 0]

y_test = y_test.cpu().numpy()
y_test = y_test.reshape(-1, 1)
y_test = np.concatenate((y_test, extra_feature), axis=1)
y_test = scaler.inverse_transform(y_test)
y_test = y_test[:, 0]
# print(predictions.shape, y_test.shape)

y_train = scaler.inverse_transform(train)
y_val = scaler.inverse_transform(val)
y_train = y_train[:, 0]
y_val = y_val[:, 0]
# print(y_train.shape, y_val.shape)

data_df = pd.DataFrame(data={'Prediction': predictions, 'Actual': y_test}, index=test.index[:predictions.shape[0]])
train_df = pd.DataFrame(data={'Train': y_train}, index=train.index)
val_df = pd.DataFrame(data={'Validation': y_val}, index=val.index)
# Plot the predictions and actual values
plt.figure(figsize=(12, 10))
plt.plot(data_df['Prediction'], label='Predicted', color='red')
plt.plot(data_df['Actual'], label='Actual', color='blue')
plt.title('Weather Prediction')
plt.xlabel('Time Step')
plt.ylabel('Temperature')
plt.legend()
# plt.savefig(setting.images_path + 'weather_lstm_rh.png')
plt.savefig(setting.images_path + 'weather_transformer_temp.png')
plt.show()

def smape(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    """
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denominator
    diff[denominator == 0] = 0  # tránh chia cho 0
    return 100 * np.mean(diff)

def mape(y_true, y_pred):
    """
    Mean Absolute Percentage Error
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # tránh chia cho 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def evaluate_model(y_true, y_pred, name=""):
    """
    Đánh giá mô hình với các metric phổ biến trong dự báo thời tiết
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape_val = mape(y_true, y_pred)
    smape_val = smape(y_true, y_pred)

    print(f"Kết quả đánh giá cho mô hình: {name}")
    print(f"MAE   : {mae:.4f}")
    print(f"RMSE  : {rmse:.4f}")
    print(f"R²    : {r2:.4f}")
    print(f"MAPE  : {mape_val:.2f}%")
    print(f"SMAPE : {smape_val:.2f}%")
    
    # return {
    #     "MAE": mae,
    #     "RMSE": rmse,
    #     "R2": r2,
    #     "MAPE": mape_val,
    #     "SMAPE": smape_val
    # }

# evaluate_model(y_test, predictions, name="LSTM")
evaluate_model(y_test, predictions, name="Transformer")
