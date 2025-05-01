import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src import setting
from src.model.WeatherLSTM import WeatherLSTM
from src.model.WeatherTransformer import WeatherTransformer
from src.split_data import create_inout_sequences_hourly, create_inout_sequences_daily

st.set_page_config(
    page_title="Weather Forecasting", 
    page_icon=":sunny:",)

@st.cache_data
def data_process(path='data/plot_data.csv'):
    df = pd.read_csv(path, skiprows=2)
    df.dropna(inplace=True)
    pd.save_csv(df, 'data/new_data.csv', index=False)
    return df
def load_data(scaler_hourly, scaler_daily, path='data/new_data.csv'):
    df_hourly = pd.read_csv(path)
    df_daily = df_hourly.copy()
    df_daily['time'] = pd.to_datetime(df_daily['time'])
    df_daily['time'] = df_daily['time'].dt.date
    df_daily = df_daily.groupby('time').agg(
        temperature_2m_mean = ('temperature_2m (°C)', 'mean'),
        relative_humidity_2m_mean = ('relative_humidity_2m (%)', 'mean'),
        temperature_2m_min = ('temperature_2m (°C)', 'min'),
        temperature_2m_max = ('temperature_2m (°C)', 'max')
    ).reset_index()

    df_daily.rename(columns={
        'temperature_2m_mean': 'temperature_2m_mean (°C)',
        'relative_humidity_2m_mean': 'relative_humidity_2m_mean (%)',
        'temperature_2m_min': 'temperature_2m_min (°C)',
        'temperature_2m_max': 'temperature_2m_max (°C)'
    }, inplace=True)

    df_daily = df_daily.round(1)
    df_hourly['time'] = pd.to_datetime(df_hourly['time'], format='ISO8601')
    df_hourly = df_hourly.set_index('time')
    df_hourly = df_hourly.sort_index(ascending=True)
    
    df_daily['time'] = pd.to_datetime(df_daily['time'], format='%Y-%m-%d')
    df_daily = df_daily.set_index('time')
    df_daily = df_daily.sort_index(ascending=True)
    
    hourly = scaler_hourly.transform(df_hourly)
    daily = scaler_daily.transform(df_daily)
    
    hourly = pd.DataFrame(hourly, columns=df_hourly.columns, index=df_hourly.index)
    daily = pd.DataFrame(daily, columns=df_daily.columns, index=df_daily.index)
    return hourly, daily
    
current_time = datetime.now()

scaler_daily = setting.scaler_daily
scaler_hourly = setting.scaler_hourly

learning_rate = 0.0001  # Learning rate for the optimizer
batch_size = 64  # Number of samples per batch
seq_length = 60  # Length of the input sequence
hidden_size = 64  # Number of features in the hidden state
num_layers = 2  # Number of recurrent layers
d_model = 64  # Dimension of the model for Transformer
num_heads = 2  # Number of attention heads for Transformer
num_layers_transformer = 2  # Number of layers for Transformer
dropout = 0.3  # Dropout rate for Transformer
input_daily_size = 4  # Number of features in the input data
output_daily_size = 4  # Number of features in the output data
input_hourly_size = 2  # Number of features in the input data
output_hourly_size = 2  # Number of features in the output data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hourly, daily = load_data(scaler_hourly, scaler_daily, path='data/new_data.csv')

model_daily = None
model_hourly = None
box_model = st.selectbox("Select Model Type", ("LSTM", "Transformer"))
if box_model == "LSTM":
    st.session_state.model = "LSTM"
    model_daily = WeatherLSTM(input_size=input_daily_size, hidden_size=hidden_size, output_size=output_daily_size, num_layers=num_layers, dropout=dropout).to(device)
    model_hourly = WeatherLSTM(input_size=input_hourly_size, hidden_size=hidden_size, output_size=output_hourly_size, num_layers=num_layers, dropout=dropout).to(device)
    model_daily.load_state_dict(torch.load('src/model/daily/lstm.pth', map_location=device))
    model_hourly.load_state_dict(torch.load('src/model/hourly/lstm.pth', map_location=device))
elif box_model == "Transformer":
    st.session_state.model = "Transformer"
    model_daily = WeatherTransformer(input_size=input_daily_size, d_model=d_model, nhead=num_heads, num_layers=num_layers_transformer, output_size=output_daily_size, dropout=dropout).to(device)
    model_hourly = WeatherTransformer(input_size=input_hourly_size, d_model=d_model, nhead=num_heads, num_layers=num_layers_transformer, output_size=output_hourly_size, dropout=dropout).to(device)
    model_daily.load_state_dict(torch.load('src/model/daily/transformer.pth', map_location=device))
    model_hourly.load_state_dict(torch.load('src/model/hourly/transformer.pth', map_location=device))
    
def get_input_seq(hourly_data, daily_data, seq_length=seq_length):
    # Lấy dữ liệu đầu vào cho mô hình    
    input_hourly_seq = hourly_data.values[-seq_length:]
    input_daily_seq = daily_data.values[-seq_length:]
    return input_hourly_seq, input_daily_seq

# Giờ cuối cùng trong data
last_time = hourly.index[-1]
target_end_time = pd.Timestamp.now().normalize() + pd.Timedelta(days=8)

n_steps = int((target_end_time - last_time) / pd.Timedelta(hours=1))

input_hourly_seq, input_daily_seq = get_input_seq(hourly, daily, seq_length=seq_length)
input_hourly_seq = torch.tensor(input_hourly_seq, dtype=torch.float32).to(device)
input_daily_seq = torch.tensor(input_daily_seq, dtype=torch.float32).to(device)
input_hourly_seq = input_hourly_seq.unsqueeze(0)  # [1, 60, features]
input_daily_seq = input_daily_seq.unsqueeze(0)  # [1, 60, features]

def predict(model, input_seq, n_steps, scaler):
    predictions = []
    with torch.no_grad():
        for _ in range(n_steps):
            out = model(input_seq)
            predictions.append(out.cpu().numpy())  # lấy [features]
            
            next_input = out.unsqueeze(1)  # [1, features]
            input_seq = torch.cat((input_seq[:, 1:, :], next_input), dim=1)
    
    predictions = np.concatenate(predictions, axis=0)  # [n_steps, features]
    predictions = scaler.inverse_transform(predictions)
    return predictions

def save_predictions(last_time, model, input_seq, n_steps, scaler, filename):
    # Giả sử predictions là np.array với shape (n_steps, 2)
    # last_time là index cuối của dữ liệu gốc
    predictions = predict(model, input_seq, n_steps, scaler)
    timestamps = [last_time + pd.Timedelta(hours=i+1) for i in range(len(predictions))]
    df_forecast = pd.DataFrame(predictions, columns=['temperature_2m (°C)', 'relative_humidity_2m (%)'], index=pd.to_datetime(timestamps))

    df = pd.DataFrame(predictions, columns=['temperature_2m (°C)', 'relative_humidity_2m (%)'])
    df.to_csv(filename, index=False)
    # st.download_button(label="Download Predictions", data=df.to_csv(index=False), file_name=filename, mime='text/csv')

if st.button("Predict Daily"):
    # Dự đoán dữ liệu hàng ngày
    n_steps_day = int((target_end_time - last_time) / pd.Timedelta(days=1))
    daily_predictions = predict(model_daily, input_daily_seq, n_steps_day, scaler_daily)
    daily_timestamps = [last_time + pd.Timedelta(days=i+1) for i in range(len(daily_predictions))]
    df_daily_forecast = pd.DataFrame(daily_predictions, columns=['temperature_2m_mean (°C)', 'relative_humidity_2m_mean (%)', 'temperature_2m_min (°C)', 'temperature_2m_max (°C)'], index=pd.to_datetime(daily_timestamps))
    df_daily_forecast['time'] = df_daily_forecast.index.date
    df_daily_forecast['time'] = pd.to_datetime(df_daily_forecast['time'])
    df_daily_forecast.to_csv('data/daily_forecast.csv', index=False)
    st.download_button(label="Download Daily Forecast", data=df_daily_forecast.to_csv(index=False), file_name='data/daily_forecast.csv', mime='text/csv')
elif st.button("Predict Hourly"):
    # Dự đoán dữ liệu hàng giờ
    hourly_predictions = predict(model_hourly, input_hourly_seq, n_steps, scaler_hourly)
    hourly_timestamps = [last_time + pd.Timedelta(hours=i+1) for i in range(len(hourly_predictions))]
    df_hourly_forecast = pd.DataFrame(hourly_predictions, columns=['temperature_2m (°C)', 'relative_humidity_2m (%)'], index=pd.to_datetime(hourly_timestamps))
    df_hourly_forecast['time'] = df_hourly_forecast.index
    df_hourly_forecast['time'] = pd.to_datetime(df_hourly_forecast['time'], format='ISO8601')
    df_hourly_forecast = df_hourly_forecast.set_index('time')
    df_hourly_forecast = df_hourly_forecast.sort_index(ascending=True)
    df_hourly_forecast.to_csv('data/hourly_forecast.csv', index=True)
    st.download_button(label="Download Hourly Forecast", data=df_hourly_forecast.to_csv(index=True), file_name='data/hourly_forecast.csv', mime='text/csv')
# save_predictions(last_time, model_hourly, input_hourly_seq, n_steps, scaler_hourly, 'data/hourly_forecast.csv')

# Dummy data
current_temp = 29
current_humidity = 48
weather_status = "Sunny"
high_temp = 31
low_temp = 19

forecast_hours = {
    "Now": (29, 45),
    "15h": (29, 44),
    "16h": (30, 42),
    "17h": (30, 41),
    "18h": (30, 40),
    "19h": (29, 42),
}

forecast_days = {
    "Today": (19, 31, 48),
    "Tomorrow": (17, 32, 45),
}

# Title
st.markdown(f"<div class='big-font'>{current_temp}°</div>", unsafe_allow_html=True)
st.markdown(f"<div class='medium-font'>{weather_status}</div>", unsafe_allow_html=True)
st.markdown(f"<div class='small-font'>H: {high_temp}°  L: {low_temp}° | Humidity: {current_humidity}%</div>", unsafe_allow_html=True)

st.markdown("---")

# Severe Weather Warning
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("⚠️ Severe Weather")
    st.write("High disruption due to extreme high temperatures. Stay hydrated and avoid outdoor activities.")
    st.markdown("</div>", unsafe_allow_html=True)

# Hourly Forecast
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("🕒 Hourly Forecast")
    cols = st.columns(len(forecast_hours))
    for idx, (hour, (temp, hum)) in enumerate(forecast_hours.items()):
        with cols[idx]:
            st.write(hour)
            st.write(f"🌡️ {temp}°C")
            st.write(f"💧{hum}%")
    st.markdown("</div>", unsafe_allow_html=True)

# 2-Day Forecast
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📅 2-Day Forecast")
    for day, (low, high, hum) in forecast_days.items():
        st.write(f"**{day}**: 🌡️ {low}°C - {high}°C | 💧 {hum}% Humidity")
    st.markdown("</div>", unsafe_allow_html=True)