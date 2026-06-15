import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# PyTorch Custom Temporal Attention Layer
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(TemporalAttention, self).__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # lstm_output shape: [batch, seq_len, hidden_dim]
        # fc input: [batch * seq_len, hidden_dim]
        # scores shape: [batch, seq_len, 1]
        scores = self.fc(lstm_output)
        weights = torch.softmax(scores, dim=1) # [batch, seq_len, 1]
        # Context vector: weighted average of sequence states
        context = torch.sum(lstm_output * weights, dim=1) # [batch, hidden_dim]
        return context, weights

# Proposed Hybrid Model Architecture (CNN-BiLSTM-Attention)
class HybridModel(nn.Module):
    def __init__(self, input_dim, cnn_channels=64, lstm_hidden=64):
        super(HybridModel, self).__init__()
        # 1D CNN: Input shape is [batch, input_dim, seq_len]
        self.conv = nn.Conv1d(in_channels=input_dim, out_channels=cnn_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.2)
        
        # BiLSTM: Input shape is [batch, seq_len, cnn_channels]
        self.lstm = nn.LSTM(input_size=cnn_channels, hidden_size=lstm_hidden, 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.2)
        
        # Attention over BiLSTM output (hidden states are bidirectional, so hidden_dim is doubled)
        self.attention = TemporalAttention(lstm_hidden * 2)
        
        # Regression network
        self.fc1 = nn.Linear(lstm_hidden * 2, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Input x shape: [batch, seq_len, input_dim]
        # Permute to match PyTorch Conv1d input: [batch, input_dim, seq_len]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Permute back to [batch, seq_len, cnn_channels]
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = self.dropout2(lstm_out)
        
        # Attention
        context, weights = self.attention(lstm_out)
        
        # Fully Connected Layers
        out = torch.relu(self.fc1(context))
        out = self.fc2(out)
        return out

# PyTorch Standalone LSTM Baseline
class StandaloneLSTM(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64):
        super(StandaloneLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # Input x shape: [batch, seq_len, 1]
        lstm_out, _ = self.lstm(x)
        # Select last timestep output
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        return out

# Statistical Seasonal Decomposition Baseline
class SeasonalBaseline:
    def __init__(self):
        self.mapping = {}
        
    def fit(self, df, target_col):
        temp_df = df.copy()
        temp_df['hour'] = temp_df.index.hour
        temp_df['dayofweek'] = temp_df.index.dayofweek
        
        self.mapping = temp_df.groupby(['dayofweek', 'hour'])[target_col].mean().to_dict()
        self.global_mean = temp_df[target_col].mean()
        
    def predict_season(self, datetimes):
        preds = []
        for dt in datetimes:
            day = dt.dayofweek
            hr = dt.hour
            preds.append(self.mapping.get((day, hr), self.global_mean))
        return np.array(preds)
