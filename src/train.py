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
from src.model.WeatherLSTM import WeatherLSTM
from src.model.WeatherTransformer import WeatherTransformer
from src.predict_utils import add_features

warnings.filterwarnings("ignore")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seed(42)

def create_inout_sequences_hourly(data, seq_length):
    # Add features first
    data = add_features(data, is_daily=False)
    
    # Select features for input
    feature_columns = ['temperature_2m (°C)', 'relative_humidity_2m (%)', 'hour_sin', 'hour_cos', 
                      'temperature_2m (°C)_rolling_mean', 'relative_humidity_2m (%)_rolling_mean',
                      'temperature_2m (°C)_rolling_std', 'relative_humidity_2m (%)_rolling_std',
                      'temperature_2m (°C)_lag_1', 'relative_humidity_2m (%)_lag_1',
                      'temp_humidity_interaction']
    
    # Select target columns
    target_columns = ['temperature_2m (°C)', 'relative_humidity_2m (%)']
    
    # Check for NaN values and handle them
    X_data = data[feature_columns].fillna(method='ffill').fillna(method='bfill').values
    y_data = data[target_columns].fillna(method='ffill').fillna(method='bfill').values
    
    # Check for infinite values
    X_data = np.nan_to_num(X_data, nan=0.0, posinf=1e6, neginf=-1e6)
    y_data = np.nan_to_num(y_data, nan=0.0, posinf=1e6, neginf=-1e6)
    
    X, y = [], []
    L = len(data)
    for i in range(seq_length, L):
        train_seq = X_data[i-seq_length:i]
        train_label = y_data[i]
        X.append(train_seq)
        y.append(train_label)
    
    X, y = np.array(X), np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

def create_inout_sequences_daily(data, seq_length):
    # Add features first
    data = add_features(data, is_daily=True)
    
    # Select features for input
    feature_columns = ['temperature_2m_mean (°C)', 'relative_humidity_2m_mean (%)', 
                      'temperature_2m_max (°C)', 'temperature_2m_min (°C)',
                      'month_sin', 'month_cos', 'day_sin', 'day_cos',
                      'temp_range', 'temperature_2m_mean (°C)_rolling_mean', 
                      'relative_humidity_2m_mean (%)_rolling_mean',
                      'temperature_2m_max (°C)_rolling_mean', 
                      'temperature_2m_min (°C)_rolling_mean',
                      'temperature_2m_mean (°C)_rolling_std', 
                      'relative_humidity_2m_mean (%)_rolling_std',
                      'temperature_2m_mean (°C)_lag_1', 
                      'relative_humidity_2m_mean (%)_lag_1',
                      'temperature_2m_mean (°C)_lag_7', 
                      'relative_humidity_2m_mean (%)_lag_7',
                      'temp_humidity_interaction']
    
    # Select target columns
    target_columns = ['temperature_2m_mean (°C)', 'relative_humidity_2m_mean (%)', 
                     'temperature_2m_max (°C)', 'temperature_2m_min (°C)']
    
    # Check for NaN values and handle them
    X_data = data[feature_columns].fillna(method='ffill').fillna(method='bfill').values
    y_data = data[target_columns].fillna(method='ffill').fillna(method='bfill').values
    
    # Check for infinite values
    X_data = np.nan_to_num(X_data, nan=0.0, posinf=1e6, neginf=-1e6)
    y_data = np.nan_to_num(y_data, nan=0.0, posinf=1e6, neginf=-1e6)
    
    X, y = [], []
    L = len(data)
    for i in range(seq_length, L):
        train_seq = X_data[i-seq_length:i]
        train_label = y_data[i]
        X.append(train_seq)
        y.append(train_label)
    
    X, y = np.array(X), np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return X_tensor, y_tensor

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
    def __init__(self, patience=80, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_model_state = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                # Restore best model state
                if self.best_model_state is not None:
                    model.load_state_dict(self.best_model_state)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, patience, device, path_model, path_image):
    best_val_loss = np.inf
    epochs_no_improve = 0
    train_losses = []
    val_losses = []
    early_stopping = EarlyStopping(patience=patience, min_delta=0.0001)
    
    # Add gradient clipping with a smaller value
    max_grad_norm = 0.5  # Reduced from 1.0 to 0.5

    # Create directory for saving models if it doesn't exist
    model_dir = os.path.dirname('/kaggle/working/model/' + path_model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    epoch_loop = tqdm(range(num_epochs), desc="Training Progress")
    for epoch in epoch_loop:
        model.train()
        total_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Check for NaN values in input
            if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                print("Warning: NaN values detected in input data")
                continue
                
            optimizer.zero_grad()
            y_pred = model(X_batch)
            
            # Check for NaN values in predictions
            if torch.isnan(y_pred).any():
                print("Warning: NaN values detected in model predictions")
                continue
                
            loss = criterion(y_pred, y_batch)
            
            # Check if loss is NaN
            if torch.isnan(loss):
                print("Warning: NaN loss detected")
                continue
                
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
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
                
                # Check for NaN values in validation data
                if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                    continue
                    
                y_pred = model(X_batch)
                
                # Check for NaN values in validation predictions
                if torch.isnan(y_pred).any():
                    continue
                    
                val_loss = criterion(y_pred, y_batch)
                
                # Check if validation loss is NaN
                if torch.isnan(val_loss):
                    continue
                    
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save the model
            save_model(model, path_model)
            
        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        epoch_loop.set_postfix(train_loss=avg_train_loss, val_loss=avg_val_loss)
        
        # Early stopping check
        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best validation loss: {early_stopping.best_loss:.4f}")

    # Plot training history
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
    num_epochs = 200
    batch_size = 32
    patience = 20

    # Train with daily data
    train_daily_path = setting.train_daily_path
    val_daily_path = setting.val_daily_path
    seq_length_daily = 30  # 30 days sequence length for daily data
    train_daily_loader, val_daily_loader = create_dataloader(train_daily_path, val_daily_path, seq_length_daily, batch_size)
    num_training_daily_steps = len(train_daily_loader) * num_epochs
    path_daily_image = setting.images_daily_path

    # Train with hourly data
    train_hourly_path = setting.train_hourly_path
    val_hourly_path = setting.val_hourly_path
    seq_length_hourly = 120  # 5 days (120 hours) sequence length for hourly data
    train_hourly_loader, val_hourly_loader = create_dataloader(train_hourly_path, val_hourly_path, seq_length_hourly, batch_size)
    num_training_hourly_steps = len(train_hourly_loader) * num_epochs
    path_hourly_image = setting.images_hourly_path

        
    print(f"Training LSTM model on daily data...")
    LSTM_model = WeatherLSTM(input_size=param_lstm['input_size'][1],
                            hidden_size=param_lstm['hidden_size'], 
                            output_size=param_lstm['output_size'][1], 
                            num_layers=param_lstm['num_layers'], 
                            dropout=param_lstm['dropout']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(LSTM_model.parameters(), lr=param_lstm['learning_rate'])
    scheduler = None
    path_model = 'daily/lstm.pth'
    train_model(LSTM_model, train_daily_loader, val_daily_loader, criterion, optimizer, scheduler, num_epochs, patience, device, path_model, path_daily_image)

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
        num_warmup_steps=int(num_training_daily_steps * 0.1),  # 10% of training steps
        num_training_steps=num_training_daily_steps,
    )
    path_model = 'daily/transformer.pth'
    train_model(Transformer_model, train_daily_loader, val_daily_loader, criterion, optimizer, scheduler, num_epochs, patience, device, path_model, path_daily_image)

    print(f"Training LSTM model on hourly data...")
    LSTM_model = WeatherLSTM(input_size=param_lstm['input_size'][0], 
                            hidden_size=param_lstm['hidden_size'], 
                            output_size=param_lstm['output_size'][0], 
                            num_layers=param_lstm['num_layers'], 
                            dropout=param_lstm['dropout']).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(LSTM_model.parameters(), lr=param_lstm['learning_rate'])
    scheduler = None
    path_model = 'hourly/lstm.pth'
    train_model(LSTM_model, train_hourly_loader, val_hourly_loader, criterion, optimizer, scheduler, num_epochs, patience, device, path_model, path_hourly_image)

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
        num_warmup_steps=int(num_training_hourly_steps * 0.1),  # 10% of training steps
        num_training_steps=num_training_hourly_steps,
    )
    path_model = 'hourly/transformer.pth'
    train_model(Transformer_model, train_hourly_loader, val_hourly_loader, criterion, optimizer, scheduler, num_epochs, patience, device, path_model, path_hourly_image)