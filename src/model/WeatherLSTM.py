import torch.nn as nn
import torch.nn.functional as F

class WeatherLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=1, num_layers=1, dropout=0.3):
        super(WeatherLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)

        # LSTM layer
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # Fully connected layer
        self.fc1 = nn.Linear(hidden_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, output_size)

    def forward(self, x):
        out, _ = self.lstm1(x)  # LSTM layer 1
        out, _ = self.lstm2(out)
        
        out = out[:, -1, :]

        # Decode the hidden state of the last time step
        out = self.fc1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out