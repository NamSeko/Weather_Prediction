import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px

from src import setting
from src.model.WeatherLSTM import WeatherLSTM
from src.model.WeatherTransformer import WeatherTransformer

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
# input_daily_size = 4  # Number of features in the input data
# output_daily_size = 4  # Number of features in the output data
input_size = 2  # Number of features in the input data
output_size = 2  # Number of features in the output data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="Weather Forecasting", 
    page_icon=":sunny:",
    # layout="wide",
    initial_sidebar_state="expanded",
    )

st.title("Weather Forecasting")
st.subheader("Predicting weather data using LSTM and Transformer models")

@st.cache_data
def split_data(path='data/new_data.csv'):
    df_hourly = pd.read_csv(path)
    df_daily = df_hourly.copy()
    df_daily['time'] = pd.to_datetime(df_daily['time'], format='ISO8601')
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
    
    return df_hourly, df_daily

def load_data(scaler_hourly, scaler_daily):
    df_hourly, df_daily = split_data(path='data/new_data.csv')
    
    hourly = scaler_hourly.transform(df_hourly)
    daily = scaler_daily.transform(df_daily)
    
    hourly = pd.DataFrame(hourly, columns=df_hourly.columns, index=df_hourly.index)
    daily = pd.DataFrame(daily, columns=df_daily.columns, index=df_daily.index)
    return hourly, daily

def take_file_input(hourly=None, daily=None):
    uploaded_file = st.file_uploader("Tải lên file CSV 📂", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, skiprows=2)
        df.dropna(inplace=True)
        df.to_csv('data/new_data.csv', index=False)
        # pd.save_csv(df, 'data/new_data.csv', index=False)
        df['time'] = pd.to_datetime(df['time'], format='ISO8601')
        df = df.set_index('time')
        df = df.sort_index(ascending=True)
        df.to_csv('data/update_data.csv', index=True)
        st.success("File uploaded successfully!")
    else:
        st.info("No file uploaded.")

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write("Upload your CSV file containing hourly weather data. The file should contain the following columns:")
st.write("- `time`: Timestamp in ISO8601 format")
st.write("- `temperature_2m (°C)`: Temperature in degrees Celsius")
st.write("- `relative_humidity_2m (%)`: Relative humidity in percentage")
st.markdown("</div>", unsafe_allow_html=True)

take_file_input()

# Kiểm tra xem đã load data sang file new_data.csv chưa
if os.path.exists('data/new_data.csv'):
    hourly, daily = load_data(scaler_hourly, scaler_daily)
else:
    st.warning("Please upload a CSV file first.")
    st.stop()

model = None
box_model = st.selectbox("Select Model Type", ("Choose one Model", "LSTM", "Transformer"))
if box_model == "LSTM":
    st.session_state.model = "LSTM"
    # model_daily = WeatherLSTM(input_size=input_daily_size, hidden_size=hidden_size, output_size=output_daily_size, num_layers=num_layers, dropout=dropout).to(device)
    # model_daily.load_state_dict(torch.load('src/model/daily/lstm.pth', map_location=device))
    model = WeatherLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout).to(device)
    model.load_state_dict(torch.load('src/model/hourly/lstm.pth', map_location=device))
elif box_model == "Transformer":
    st.session_state.model = "Transformer"
    # model_daily = WeatherTransformer(input_size=input_daily_size, d_model=d_model, nhead=num_heads, num_layers=num_layers_transformer, output_size=output_daily_size, dropout=dropout).to(device)
    # model_daily.load_state_dict(torch.load('src/model/daily/transformer.pth', map_location=device))
    model = WeatherTransformer(input_size=input_size, d_model=d_model, nhead=num_heads, num_layers=num_layers_transformer, output_size=output_size, dropout=dropout).to(device)
    model.load_state_dict(torch.load('src/model/hourly/transformer.pth', map_location=device))
else:
    model = None
    
def get_input_seq(hourly_data, seq_length=seq_length):
    # Lấy dữ liệu đầu vào cho mô hình    
    input_seq = hourly_data.values[-seq_length:]
    # input_daily_seq = daily_data.values[-seq_length:]
    return input_seq

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

def update_data(path='data/update_data.csv', df_forecast=None):
    # Cập nhật dữ liệu mới vào file CSV
    if df_forecast is not None:
        # Đọc file cũ
        try:
            df_old = pd.read_csv(path, index_col=0, parse_dates=True)
            df_old.index = df_old.index.strftime('%Y-%m-%dT%H:%M')
        except FileNotFoundError:
            df_old = pd.DataFrame()  # nếu file chưa tồn tại, tạo DataFrame rỗng

        # Ghép dữ liệu mới vào
        df_forecast = df_forecast.round(1)
        df_combined = pd.concat([df_old, df_forecast])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]  # giữ giá trị mới

        # Ghi đè file
        df_combined.to_csv(path, index=True, float_format='%.1f')
    
def plot_forecast(df, title):
    fig = px.line(df, 
                  x=df.index, 
                  y=['temperature_2m (°C)', 'relative_humidity_2m (%)'],
                  markers=False, title=title)
    fig.update_traces(mode='lines')
    fig.update_layout(xaxis_title='Time', yaxis_title='Value', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)


if model is not None:
    # Giờ cuối cùng trong data
    last_time = hourly.index[-1]
    last_time = pd.to_datetime(last_time, format='ISO8601')
    target_end_time = pd.Timestamp.now().normalize() + pd.Timedelta(days=8)

    n_steps = int((target_end_time - last_time) / pd.Timedelta(hours=1))

    input_seq = get_input_seq(hourly, seq_length=seq_length)
    input_seq = torch.tensor(input_seq, dtype=torch.float32).to(device)
    # input_daily_seq = torch.tensor(input_daily_seq, dtype=torch.float32).to(device)
    input_seq = input_seq.unsqueeze(0)  # [1, 60, features]
    # input_daily_seq = input_daily_seq.unsqueeze(0)  # [1, 60, features]
    
    # Dự đoán dữ liệu hàng giờ
    predictions = predict(model, input_seq, n_steps, scaler_hourly)

    timestamps = [last_time + pd.Timedelta(hours=i+1) for i in range(len(predictions))]
    df_forecast = pd.DataFrame(predictions, columns=['temperature_2m (°C)', 'relative_humidity_2m (%)'], index=pd.to_datetime(timestamps))
    df_forecast['time'] = df_forecast.index
    df_forecast['time'] = pd.to_datetime(df_forecast['time'], format='ISO8601')
    df_forecast = df_forecast.set_index('time')
    df_forecast = df_forecast.sort_index(ascending=True)
    df_forecast = df_forecast.round(1)

    plot_forecast(df_forecast, 'Weather Forecast')

    df_forecast.index = pd.to_datetime(df_forecast.index, format='ISO8601')
    df_forecast.index = df_forecast.index.strftime('%Y-%m-%dT%H:%M')
    update_data(path='data/update_data.csv', df_forecast=df_forecast)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    df_forecast.to_csv('data/weather_forecast.csv', index=True)
    st.download_button(label="Download Weather Forecast", data=df_forecast.to_csv(index=True), file_name='data/weather_forecast.csv', mime='text/csv')
    st.markdown("</div>", unsafe_allow_html=True)
        
    df_hourly, df_daily = split_data(path='data/update_data.csv')

    # Hiển thị list các ngày trong dự đoán (có thể chọn)
    today_idx = df_daily.index.date.tolist().index(current_time.date())
    limited_dates = df_daily.index.date[today_idx-7:today_idx + 8]  # Chỉ lấy 7 ngày tiếp theo
    today_idx = limited_dates.tolist().index(current_time.date())
    selected_date = st.selectbox("Select Date", limited_dates, index=today_idx)
    # Sau khi chọn ngày, hiển thị dự đoán cho ngày đó với nhiệt độ, độ ẩm trung bình và min, max và nhiệt độ, độ ẩm từng giờ của ngày đó
    # st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write(f"Weather forecast for {selected_date}:")
    df_daily_selected = df_daily[df_daily.index.date == selected_date]
    df_hourly_selected = df_hourly[df_hourly.index.date == selected_date]    
    
    def get_weather_icon(temp_c, humidity):
        if humidity > 90 and temp_c < 26:
            return "🌧️"
        elif 85 <= humidity <= 90 and 26 <= temp_c <= 28:
            return "🌦️"
        elif temp_c >= 30 and humidity > 70:
            return "☀️"
        elif 28 <= temp_c <= 32 and 70 < humidity <= 85:
            return "⛅"
        else:
            return "🌤️"
        
    opacity = None
    if selected_date >= current_time.date():
        opacity = 1.0
    else:
        opacity = 0.5
        
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 20px; font-weight: bold; opacity: {opacity};">
            <div style="margin-bottom: 10px;">
                <span>
                    {get_weather_icon(df_daily_selected['temperature_2m_mean (°C)'].values[0], df_daily_selected['relative_humidity_2m_mean (%)'].values[0])}
                </span>
                <span style="color: #FF5733;">
                    {df_daily_selected['temperature_2m_mean (°C)'].values[0]}°C
                </span>
            </div>
            <div>
                <span style="margin-right: 20px;">
                    <b>H</b>: {df_daily_selected['temperature_2m_max (°C)'].values[0]}°C
                </span>
                <span style="margin-left: 20px;">
                    <b>L</b>: {df_daily_selected['temperature_2m_min (°C)'].values[0]}°C
                </span>
            </div>
            <div style="margin-top: 10px; color: blue;">
                {df_daily_selected['relative_humidity_2m_mean (%)'].values[0]}%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    # Hiển thị dự đoán hàng giờ theo từng giờ với nhiệt độ và độ ẩm
    for hour, temp, humidity in zip(df_hourly_selected.index, df_hourly_selected['temperature_2m (°C)'], df_hourly_selected['relative_humidity_2m (%)']):
        hour = int(hour.strftime('%H'))
        if selected_date > current_time.date():
            opacity = 1.0
        elif selected_date == current_time.date():
            if hour < current_time.hour:
                opacity = 0.5
            else:
                opacity = 1.0
        else:
                opacity = 0.5
        icon = get_weather_icon(temp, humidity)
        st.markdown(
            f"""
            <div style="text-align: center; font-size: 15px; font-weight: bold; opacity: {opacity};">
                <div style=" margin-bottom: 10px;">
                    <span style="font-size: 20px;">{icon}</span>
                    <span style="margin-right: 20px;">{hour}h</span>
                    <span style="margin-right: 20px; color: red;">{temp}°C</span>
                    <span style="color: blue;">{humidity}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    plot_forecast(df_hourly_selected, 'Hourly Weather Forecast')