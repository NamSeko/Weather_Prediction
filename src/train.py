import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm

from src import setting
from src.split_data import create_inout_sequences_hourly, create_inout_sequences_daily
from src.model.WeatherLSTM import WeatherLSTM
from src.model.WeatherTransformer import WeatherTransformer

warnings.filterwarnings("ignore")

def create_dataloader(train_path, val_path, seq_length, batch_size):
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    
    if 'daily' in train_path:
        X_train, y_train = create_inout_sequences_daily(train_data, seq_length)
        X_val, y_val = create_inout_sequences_daily(val_data, seq_length)
    elif 'hourly' in train_path:
        X_train, y_train = create_inout_sequences_hourly(train_data, seq_length)
        X_val, y_val = create_inout_sequences_hourly(val_data, seq_length)

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def save_model(model, path):
    """
    Save the model to a file.

    Args:
        model (torch.nn.Module): The model to save.
        path (str): The path to save the model.
    """
    torch.save(model.state_dict(), './src/model/' + path)
    print(f"Model saved !")
    
# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, path_model, path_image):
    train_loss = []
    val_history_loss = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            y_pred = model(X_batch)
            # loss = criterion(y_pred, y_batch.view(-1, 1))
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        total_loss /= len(train_loader)
        train_loss.append(total_loss)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                y_pred = model(X_batch)
                # val_loss += criterion(y_pred, y_batch.view(-1, 1)).item()
                val_loss += criterion(y_pred, y_batch).item()

        val_loss /= len(val_loader)
        val_history_loss.append(val_loss)
        # print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}')

    min_loss = min(val_history_loss)
    print(f"Loss: {min_loss:.4f}")
    print("Finish training!!!")
    # Save the model
    save_model(model, path_model)

    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_history_loss, label='Validation Loss')
    plt.title('Train and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if not os.path.exists(path_image):
        os.makedirs(path_image)
    
    name_image = path_model.replace('.pth', '.png')
    name_image = name_image.replace('/', '_')
    plt.savefig(path_image+name_image)
    
if __name__ == "__main__":   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Define hyperparameters
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
    
    # Train with daily data
    train_daily_path = setting.train_daily_path
    val_daily_path = setting.val_daily_path
    scaler = setting.scaler_daily
    path_daily_image = setting.images_daily_path
    
    df = pd.read_csv(train_daily_path)
    columns = df.columns[1:]
    input_size = len(columns)  # Number of features in the input data
    output_size = len(columns)  # Number of features in the output data
    
    train_loader, val_loader = create_dataloader(train_daily_path, val_daily_path, seq_length, batch_size)
    
    print(f"Training LSTM model on daily data...")
    LSTM_model = WeatherLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(LSTM_model.parameters(), lr=learning_rate)
    path_model = 'daily/lstm.pth'
    train_model(LSTM_model, train_loader, val_loader, criterion, optimizer, num_epochs, path_model, path_daily_image)
    
    print(f"Training Transformer model on daily data...")
    Transformer_model = WeatherTransformer(input_size=input_size, d_model=d_model, nhead=num_heads, num_layers=num_layers_transformer, output_size=output_size, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(Transformer_model.parameters(), lr=learning_rate)
    path_model = 'daily/transformer.pth'
    train_model(Transformer_model, train_loader, val_loader, criterion, optimizer, num_epochs, path_model, path_daily_image)
    
    # Train with hourly data
    train_hourly_path = setting.train_hourly_path
    val_hourly_path = setting.val_hourly_path
    scaler = setting.scaler_hourly
    path_hourly_image = setting.images_hourly_path
    
    df = pd.read_csv(train_hourly_path)
    columns = df.columns[1:]
    input_size = len(columns)  # Number of features in the input data
    output_size = len(columns)  # Number of features in the output data
    
    train_loader, val_loader = create_dataloader(train_hourly_path, val_hourly_path, seq_length, batch_size)
    
    print(f"Training LSTM model on hourly data...")
    LSTM_model = WeatherLSTM(input_size=input_size, hidden_size=hidden_size, output_size=output_size, num_layers=num_layers, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(LSTM_model.parameters(), lr=learning_rate)
    path_model = 'hourly/lstm.pth'
    train_model(LSTM_model, train_loader, val_loader, criterion, optimizer, num_epochs, path_model, path_hourly_image)
    
    print(f"Training Transformer model on hourly data...")
    Transformer_model = WeatherTransformer(input_size=input_size, d_model=d_model, nhead=num_heads, num_layers=num_layers_transformer, output_size=output_size, dropout=dropout).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(Transformer_model.parameters(), lr=learning_rate)
    path_model = 'hourly/transformer.pth'
    train_model(Transformer_model, train_loader, val_loader, criterion, optimizer, num_epochs, path_model, path_hourly_image)
    