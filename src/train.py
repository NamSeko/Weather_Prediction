from model.WeatherLSTM import WeatherLSTM
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt

scaler = joblib.load('../data/scaler.save')
train_data = np.load('../data/daily/temp/train_data.npy')
val_data = np.load('../data/daily/temp/val_data.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Define hyperparameters
input_size = 5  # Number of features in the input data
hidden_size = 64  # Number of features in the hidden state
num_layers = 2  # Number of recurrent layers
output_size = 1  # Number of features in the output data
learning_rate = 0.001  # Learning rate for the optimizer
batch_size = 64  # Number of samples per batch

seq_length = 30  # Length of the input sequence

num_epochs = 60  # Number of epochs to train the model

model = WeatherLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size).to(device)
# Define the model, loss function, and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def create_inout_sequences(data, seq_length):
    X, y = [], []
    L = len(data)
    for i in range(L - seq_length):
        train_seq = data[i:i + seq_length]
        train_label = data[i][0]  # Assuming the label is the first feature
        X.append(train_seq)
        y.append(train_label)
    X, y = np.array(X), np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    return X_tensor, y_tensor

X_train, y_train = create_inout_sequences(train_data, seq_length)
X_val, y_val = create_inout_sequences(val_data, seq_length)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
val_dataset = torch.utils.data.TensorDataset(X_val, y_val)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training loop
train_loss = []
val_history_loss = []
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i, (X_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch.view(-1, 1))
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
            val_loss += criterion(y_pred, y_batch.view(-1, 1)).item()

    val_loss /= len(val_loader)
    val_history_loss.append(val_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
 
# Save the model
torch.save(model.state_dict(), './model/weather_lstm.pth')

plt.figure(figsize=(12, 6))
plt.plot(train_loss, label='Train Loss')
plt.plot(val_history_loss, label='Validation Loss')
plt.title('Train and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('./images/loss_plot.png')
plt.show() 