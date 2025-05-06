import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from transformers import get_scheduler

from src import setting
from src.split_data import create_inout_sequences_hourly, create_inout_sequences_daily
from src.model.WeatherLSTM import WeatherLSTM
from src.model.WeatherTransformer import WeatherTransformer

warnings.filterwarnings("ignore")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(42)

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

# Early stopping class
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, device, path_model, path_image):
    best_val_loss = np.inf
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    epoch_loop = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in epoch_loop:
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                val_loss = criterion(y_pred, y_batch)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Update scheduler (ReduceLROnPlateau)
        # Step scheduler if using ReduceLROnPlateau
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        epoch_loop.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)
        

        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            save_model(model, path_model)
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print(f"Best validation loss: {best_val_loss:.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
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

    param_lstm = setting.param_lstm
    param_transformer = setting.param_transformer
    # Define hyperparameters
    num_epochs = 150  # Number of epochs to train the model
    seq_length = 72  # Length of the input sequence
    batch_size = 64  # Batch size for training
    patience = 20  # Early stopping patience
    # Train with daily data
    train_daily_path = setting.train_daily_path
    val_daily_path = setting.val_daily_path
    scaler = setting.scaler_daily
    path_daily_image = setting.images_daily_path
    
    train_loader, val_loader = create_dataloader(train_daily_path, val_daily_path, seq_length, batch_size)
    num_training_steps = len(train_loader) * num_epochs
        
    print(f"Training LSTM model on daily data...")
    LSTM_model = WeatherLSTM(input_size=param_lstm['input_size'][1],
                             hidden_size=param_lstm['hidden_size'], 
                             output_size=param_lstm['output_size'][1], 
                             num_layers=param_lstm['num_layers'], 
                             dropout=param_lstm['dropout']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(LSTM_model.parameters(), lr=param_lstm['learning_rate'])
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),  # 10% of training steps
        num_training_steps=num_training_steps,
    )
    path_model = 'daily/lstm.pth'
    train_model(LSTM_model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, device, path_model, path_daily_image)
    
    print(f"Training Transformer model on daily data...")
    Transformer_model = WeatherTransformer(input_size=param_transformer['input_size'][1], 
                                           d_model=param_transformer['d_model'], 
                                           nhead=param_transformer['num_head'], 
                                           num_layers=param_transformer['num_layers_transformer'], 
                                           output_size=param_transformer['output_size'][1], 
                                           dropout=param_transformer['dropout']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(Transformer_model.parameters(), lr=param_transformer['learning_rate'])
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),  # 10% of training steps
        num_training_steps=num_training_steps,
    )
    path_model = 'daily/transformer.pth'
    train_model(Transformer_model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, device, path_model, path_daily_image)
    
    # Train with hourly data
    train_hourly_path = setting.train_hourly_path
    val_hourly_path = setting.val_hourly_path
    scaler = setting.scaler_hourly
    path_hourly_image = setting.images_hourly_path
    
    train_loader, val_loader = create_dataloader(train_hourly_path, val_hourly_path, seq_length, batch_size)
    num_training_steps = len(train_loader) * num_epochs
    
    print(f"Training LSTM model on hourly data...")
    LSTM_model = WeatherLSTM(input_size=param_lstm['input_size'][0], 
                             hidden_size=param_lstm['hidden_size'], 
                             output_size=param_lstm['output_size'][0], 
                             num_layers=param_lstm['num_layers'], 
                             dropout=param_lstm['dropout']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(LSTM_model.parameters(), lr=param_lstm['learning_rate'])
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),  # 10% of training steps
        num_training_steps=num_training_steps,
    )
    path_model = 'hourly/lstm.pth'
    train_model(LSTM_model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, device, path_model, path_hourly_image)
    
    print(f"Training Transformer model on hourly data...")
    Transformer_model = WeatherTransformer(input_size=param_transformer['input_size'][0], 
                                           d_model=param_transformer['d_model'], 
                                           nhead=param_transformer['num_head'], 
                                           num_layers=param_transformer['num_layers_transformer'], 
                                           output_size=param_transformer['output_size'][0], 
                                           dropout=param_transformer['dropout']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(Transformer_model.parameters(), lr=param_transformer['learning_rate'])
    scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),  # 10% of training steps
        num_training_steps=num_training_steps,
    )
    path_model = 'hourly/transformer.pth'
    train_model(Transformer_model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, device, path_model, path_hourly_image)
    