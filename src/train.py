import torch
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import os
import matplotlib.pyplot as plt
import setting
from split_data import create_inout_sequences
import warnings

warnings.filterwarnings("ignore")

scaler = setting.scaler
train_path = setting.train_path
val_path = setting.val_path
path_image = setting.images_path


train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)

device = setting.device
print(f"Using device: {device}")


# Define hyperparameters
learning_rate = setting.learning_rate  # Learning rate for the optimizer
batch_size = setting.batch_size  # Number of samples per batch

seq_length = setting.seq_length  # Length of the input sequence

num_epochs = setting.num_epochs  # Number of epochs to train the model

# model = setting.LSTM_model.to(device)
model = setting.Transformer_model.to(device)


# Define the model, loss function, and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
setting.save_model(model)

plt.figure(figsize=(12, 6))
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
# plt.savefig(path_image+'loss_lstm_rh.png')
plt.savefig(path_image+'loss_transformer_rh.png')
plt.show() 