from torch import nn


class DeepLog(nn.Module):
    def __init__(self, input_size=36, hidden_size=128, num_layers=2, num_keys=36):
        super(DeepLog, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_keys)

    def forward(self, X):
        out, _ = self.lstm(X)
        out = self.fc(out[:, -1, :])
        return out
