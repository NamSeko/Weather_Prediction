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

model_type = st.selectbox("Select Model Type", ("LSTM", "Transformer"))

st.write(f"You selected: {model_type}")
st.write("## Hyperparameters")
st.write(f"Input Size: {input_size}")
st.write(f"Hidden Size: {hidden_size}")
st.write(f"Number of Layers: {num_layers}")
st.write(f"Output Size: {output_size}")
st.write(f"Learning Rate: {learning_rate}")
st.write(f"Batch Size: {batch_size}")
st.write(f"Sequence Length: {seq_length}")
st.write(f"Model Dimension: {d_model}")
st.write(f"Number of Heads: {num_heads}")
st.write(f"Number of Transformer Layers: {num_layers_transformer}")
st.write(f"Dropout Rate: {dropout}")
st.write(f"Device: {device}")

st.write("## Model Summary")
if model_type == "LSTM":
    model = setting.LSTM_model
    st.write(model)
elif model_type == "Transformer":
    model = setting.Transformer_model
    st.write(model)
    
st.write("## Model Training")
if st.button("Train Model"):
    # Add your training code here
    st.write("Training started...")
    # Call your training function here
    # train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    st.write("Training completed!")
    st.write("## Model Evaluation")
    # Add your evaluation code here
    st.write("Evaluation started...")
    # Call your evaluation function here
    # evaluate_model(model, test_loader)
    st.write("Evaluation completed!")
    st.write("## Model Saving")
    # Add your model saving code here
    st.write("Saving model...")
    # Call your model saving function here
    # save_model(model, path)
    st.write("Model saved!")
    st.write("## Model Loading")
    # Add your model loading code here
    st.write("Loading model...")
    # Call your model loading function here
    # load_model(model, path)
    st.write("Model loaded!")
    st.write("## Model Prediction")
    # Add your prediction code here
    st.write("Making predictions...")
    # Call your prediction function here
    # predictions = predict(model, test_loader)
    st.write("Predictions completed!")
    st.write("## Results")
    # Add your results code here
    st.write("Displaying results...")