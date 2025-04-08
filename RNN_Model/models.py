import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self, in_features=12, seq_len=8, hidden_features=32, dropout=0.0):
        super().__init__()
        
        self.input_layer = nn.Linear(in_features*seq_len, hidden_features)
        self.middle_layer = nn.Linear(hidden_features, hidden_features)
        self.output_layer = nn.Linear(hidden_features, 1)

        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.relu = nn.LeakyReLU(0.02)

    def forward(self, x):
        x = torch.flatten(x, 1)
        # Input layer
        x = self.relu(self.bn1(self.input_layer(x)))
        x = self.dropout(x)
        # Middle layer
        x = self.relu(self.bn2(self.middle_layer(x)))
        x = self.dropout(x)
        # Final output layer
        x = self.output_layer(x)
        return x.squeeze()  # Remove extra dimension for regression tasks


class BaseballRNN(torch.nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=256, n_layers=1, bidirectional=False, hidden_init='rand', rnn_dropout=0.2):
        super().__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.hidden_init = hidden_init
        self.rnn_dropout = rnn_dropout

        # Input fc layer
        self.fc_in = nn.Linear(input_size, self.hidden_size)
        # GRU Layer
        self.rnn1 = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True, num_layers=self.n_layers, dropout=self.rnn_dropout, bidirectional=False)
        # Dropout layer after last GRU layer
        self.layernorm = nn.LayerNorm(self.hidden_size)
        # Output layer
        self.fc_out = nn.Linear(self.hidden_size, output_size) 
        
        # RELU
        self.relu = nn.LeakyReLU(negative_slope=0.02)
        self.dropout = nn.Dropout(self.rnn_dropout)

    def _init_hidden(self, batch_size, device):
        """Initialize the hidden state for the GRU."""
        if self.hidden_init == 'zero':
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        elif self.hidden_init == 'rand':
            h0 = torch.randn(self.n_layers, batch_size, self.hidden_size, device=device)
        elif self.hidden_init == 'xavier':
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            nn.init.xavier_uniform_(h0)
        else: raise ValueError(f"Unsupported hidden_init: {self.hidden_init}")
        return h0

    def forward(self, x):      
        # Initial hidden states for RNNs
        h0 = self._init_hidden(x.size(0), x.device)
        
        # Pre RNN Linear Layer
        x = self.relu(self.fc_in(x))
        x = self.dropout(x)

        # Send through RNN 
        out, hidden = self.rnn1(x, h0)
        # Get output of the last time step
        out = out[:, -1, :]  # (batch_size, seq_len, hidden_size)
        out = self.layernorm(out)

        # Output layer
        out = self.relu(self.fc_out(out))

        return out.squeeze()
    

