import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import torch
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import gdown
import warnings
warnings.filterwarnings("ignore")

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

param_lstm = setting.param_lstm
param_transformer = setting.param_transformer

seq_length = 120  # Length of the input sequence
batch_size = 32  # Batch size for training

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

def download_model_if_not_exists(file_path, gdrive_id):
    if not os.path.exists(file_path):
        print(f"👉 Model chưa có, đang tải từ Google Drive...")
        url = f'https://drive.google.com/uc?id={gdrive_id}'
        gdown.download(url, file_path, quiet=False)
        print("✅ Tải model xong rồi nhé!")
    else:
        print("✅ Model đã có sẵn, không cần tải lại.")
        
model_path = 'lstm.pth'
gdrive_file_id = '1GoZosMc81iXtxxIA-9ecPT0N4bYAEPef'
download_model_if_not_exists(model_path, gdrive_file_id)

model_path = 'transformer.pth'
gdrive_file_id = '1RHCdRiN7qH25ausNjkoZ0pMycgFuNzQR'
download_model_if_not_exists(model_path, gdrive_file_id)

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
    model = WeatherLSTM(input_size=param_lstm['input_size'][0], 
                        hidden_size=param_lstm['hidden_size'], 
                        output_size=param_lstm['output_size'][0], 
                        num_layers=param_lstm['num_layers'], 
                        dropout=param_lstm['dropout']).to(device)
    # model.load_state_dict(torch.load('src/model/hourly/lstm.pth', map_location=device))
    model.load_state_dict(torch.load('/mount/src/weather_prediction/lstm.pth', map_location='cpu'))
elif box_model == "Transformer":
    st.session_state.model = "Transformer"
    model = WeatherTransformer(input_size=param_transformer['input_size'][0], 
                               d_model=param_transformer['d_model'], 
                               nhead=param_transformer['num_head'], 
                               num_layers=param_transformer['num_layers_transformer'], 
                               output_size=param_transformer['output_size'][0], 
                               dropout=param_transformer['dropout']).to(device)
    # model.load_state_dict(torch.load('src/model/hourly/transformer.pth', map_location=device))
    model.load_state_dict(torch.load('/mount/src/weather_prediction/transformer.pth', map_location='cpu'))
else:
    model = None
    
def get_input_seq(hourly_data, seq_length=seq_length):
    """Create input sequences with enhanced features"""
    # Add features to input data
    processed_data = add_features(hourly_data.reset_index(), is_daily=False)
    
    # Select features for hourly prediction - match exactly with training features
    feature_columns = [
        'temperature_2m (°C)', 'relative_humidity_2m (%)', 
        'hour_sin', 'hour_cos',
        'temperature_2m (°C)_rolling_mean', 'relative_humidity_2m (%)_rolling_mean',
        'temperature_2m (°C)_rolling_std', 'relative_humidity_2m (%)_rolling_std',
        'temperature_2m (°C)_lag_1', 'relative_humidity_2m (%)_lag_1',
        'temp_humidity_interaction'
    ]
    
    # Check for NaN values and handle them
    X_data = processed_data[feature_columns].fillna(method='ffill').fillna(method='bfill').values
    X_data = np.nan_to_num(X_data, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Create sequences for prediction
    sequences = []
    for i in range(len(X_data) - seq_length + 1):
        sequences.append(X_data[i:i + seq_length])
    
    return np.array(sequences)

def predict_sequence(model, input_seq, n_steps, scaler):
    """Generate predictions using sequence-to-sequence approach"""
    predictions = []
    current_sequence = torch.tensor(input_seq[-1:], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        for _ in range(n_steps):
            # Get prediction for next step
            pred = model(current_sequence)
            predictions.append(pred[0].cpu().numpy())
            
            # Create new sequence for next prediction
            new_sequence = torch.zeros_like(current_sequence)
            new_sequence[0, :-1] = current_sequence[0, 1:].clone()
            
            # Update only the first two features (temperature and humidity)
            new_sequence[0, -1, :2] = pred[0].clone()
            
            # Keep other features unchanged
            new_sequence[0, -1, 2:] = current_sequence[0, -1, 2:].clone()
            
            current_sequence = new_sequence
    
    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions)
    return predictions

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
    data['hour'] = data['time'].dt.hour
    
    # Add cyclic features for time
    data['month_sin'] = np.sin(2 * np.pi * data['month']/12)
    data['month_cos'] = np.cos(2 * np.pi * data['month']/12)
    data['day_sin'] = np.sin(2 * np.pi * data['day_of_year']/365)
    data['day_cos'] = np.cos(2 * np.pi * data['day_of_year']/365)
    data['hour_sin'] = np.sin(2 * np.pi * data['hour']/24)
    data['hour_cos'] = np.cos(2 * np.pi * data['hour']/24)
    
    if is_daily:
        # For daily data
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

    # Get input sequences with enhanced features
    input_sequences = get_input_seq(hourly, seq_length=seq_length)
    
    # Dự đoán dữ liệu hàng giờ với các feature mới
    predictions = predict_sequence(model, input_sequences, n_steps, scaler_hourly)

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
    
    st.write(f"Weather forecast for {selected_date}:")
    df_daily_selected = df_daily[df_daily.index.date == selected_date]
    df_hourly_selected = df_hourly[df_hourly.index.date == selected_date]    
    
    def get_weather_icon(temp, humidity):
        if temp <= 15:
            if humidity <= 40:
                return '❄️💨'
            elif humidity <= 60:
                return '❄️🌥️'
            elif humidity <= 80:
                return '❄️🌫️'
            else:
                return '❄️🌧️'
        elif temp <= 22:
            if humidity <= 40:
                return '🌬️💨'
            elif humidity <= 60:
                return '🌥️'
            elif humidity <= 80:
                return '🌥️'
            else:
                return '🌥️'
        elif temp <= 28:
            if humidity <= 40:
                return '🌤️💨'
            elif humidity <= 60:
                return '🌤️'
            elif humidity <= 90:
                return '🌤️'
            else:
                return '🌤️🌧️'
        elif temp <= 34:
            if humidity <= 40:
                return '☀️💨'
            elif humidity <= 60:
                return '☀️🌤️'
            elif humidity <= 90:
                return '☀️'
            else:
                return '☀️🌧️'
        else:
            if humidity <= 40:
                return '🔥💨'
            elif humidity <= 60:
                return '🔥☀️'
            elif humidity <= 80:
                return '🔥🌦️'
            else:
                return '🔥🌩️'
        
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
    for hour, temp, humidity in zip(df_hourly_selected.index, 
                                  df_hourly_selected['temperature_2m (°C)'], 
                                  df_hourly_selected['relative_humidity_2m (%)']):
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
