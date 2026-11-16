import torch
import torch.nn as nn

class TelemetryLSTM(nn.Module):
    """2-layer LSTM, 80 units, 0.3 dropout - exactly as in JPL paper."""
    def __init__(self, input_dim, hidden=80, layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=layers,
                            dropout=dropout if layers > 1 else 0,
                            batch_first=True)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, L, H)
        return self.fc(out[:, -1, :]).squeeze(-1)  # predict next value
