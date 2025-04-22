import streamlit as st
import sys
import torch
import os
from src import setting


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

input_size = setting.input_size
hidden_size = setting.hidden_size
num_layers = setting.num_layers
output_size = setting.output_size
learning_rate = setting.learning_rate
batch_size = setting.batch_size
seq_length = setting.seq_length
d_model = setting.d_model
num_heads = setting.num_heads
num_layers_transformer = setting.num_layers_transformer
dropout = setting.dropout
device = setting.device

st.title("Weather Forecasting")

LSTM_model = st.checkbox("LSTM model!")
Transformer_model = st.checkbox("Transformer model!")
model = None
if LSTM_model:
    from src.model.WeatherLSTM import WeatherLSTM
    model = WeatherLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)
    model.load_state_dict(torch.load('./src/model/daily/weather_lstm_temp.pth'))
    st.caption("LSTM model selected !!!")
    st.image('./src/image/daily/loss_lstm_temp.png', caption='LSTM model loss')
    st.image('./src/image/daily/weather_lstm_temp.png', caption='LSTM model architecture')
elif Transformer_model:
    from src.model.WeatherTransformer import WeatherTransformer
    model = WeatherTransformer(
        input_size=input_size,
        d_model=d_model,
        nhead=num_heads,
        num_layers=num_layers_transformer,
        output_size=output_size,
        dropout=dropout
    ).to(device)
    model.load_state_dict(torch.load('./src/model/daily/weather_transformer_temp.pth'))
    st.caption("Transformer model selected !!!")
    st.image('./src/image/daily/loss_transformer_temp.png', caption='Transformer model loss')
    st.image('./src/image/daily/weather_transformer_temp.png', caption='Transformer model architecture')
elif LSTM_model and Transformer_model:
    st.error("Please select only one model!")
    st.stop()
else:
    st.error("Please select a model!")
    st.stop()
    