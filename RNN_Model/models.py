"""
Neural network (Linear, RNN) module classes.
Last updated: 4/11/2025
"""
import torch
import torch.nn as nn


class SimpleNet(nn.Module):
    """Basic fully-connected linear neural network.

    Basic architecture is:
        input -> input linear layer -> linear layer -> output linear layer.
    Input and middle linear layers use batchnorm, LeakyReLU, and dropout:
        output -> BN -> RELU -> Dropout.
    Uses dropout and batch normalization, along with LeakyReLU activation.
    Outputs a single value for regression task.

    Function attributes:
        input_layer: Input PyTorch linear layer (input_size*seq_len {flattened} -> hidden_size).
        middle_layer: Middle PyTorch linear layer (hidden_size -> hidden_size).
        output_layer: Output PyTorch linear layer (hidden_size -> 1 {output size}).
        dropout: Dropout PyTorch layer.
        bn1: First PyTorch Batch Norm layer.
        bn2: Second PyTorch Batch Norm layer.
        relu: LeakyReLU PyTorch activation, with negative slope = 0.02.
    """
    def __init__(self, input_size=12, seq_len=8, hidden_features=32, dropout=0.2):
        """Initializes the instance based on spam preference.

        Args:
          input_size (int): Input size of model in terms of variables.
          seq_len (int): Length of sequences in input.
          hidden_features (int): Number of features in linear layers. Defaults to 32.
          dropout: Dropout percentage to use. Range in [0.0, 1.0].
        """
        super().__init__()
        
        # Linear layers
        self.input_layer = nn.Linear(input_size*seq_len, hidden_features, bias=False)
        self.middle_layer = nn.Linear(hidden_features, hidden_features, bias=False)
        self.output_layer = nn.Linear(hidden_features, 1)
        # Functional layers
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.bn2 = nn.BatchNorm1d(hidden_features)
        self.relu = nn.LeakyReLU(negative_slope=0.02)

    def forward(self, x):
        # Flatten input (variables x sequence length)
        x = torch.flatten(x, 1)
        # Input layer
        x = self.relu(self.bn1(self.input_layer(x)))
        x = self.dropout(x)
        # Middle layer
        x = self.relu(self.bn2(self.middle_layer(x)))
        x = self.dropout(x)
        # Final output layer
        x = self.relu(self.output_layer(x))
        return x.squeeze()  # Remove extra dimension for regression tasks


class BaseballRNN(torch.nn.Module):
    """RNN model using GRU cells.

    Basic architecture is:
        input -> linear layer -> GRU cells -> output layer.
    Uses dropout and layer normalization, along with LeakyReLU activation.
    Outputs a single value for regression task.

    Function attributes:
        fc_in: Input PyTorch linear layer.
        rnn1: PyTorch GRU layers
        fc_out: Output PyTorch linear layer.
        layernorm: Layer normalization PyTorch layer.
        dropout: Dropout PyTorch layer.
        relu: LeakyReLU PyTorch activation, with negative slope = 0.02..
    """
    def __init__(self, input_size, hidden_size=256, n_layers=1, bidirectional=False, hidden_init='rand', rnn_dropout=0.2):
        """Initializes the instance based on spam preference.

        Args:
          input_size (int): Input size of model.
          hidden_size (int): Hidden size in RNN layers. Defaults to 256.
          n_layers (int): Number of RNN layers to use.
          bidirectional (boolean): Whether to use bidirectional RNN.
          hidden_init (str): String indicating RNN hidden state initialization. Default to 'rand'.
          rnn_dropout (float): Dropout percentage to use. Range in [0.0, 1.0].
        """
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
        self.fc_out = nn.Linear(self.hidden_size, 1) 
        
        # RELU and non-RNN dropout
        self.relu = nn.LeakyReLU(negative_slope=0.02)
        self.dropout = nn.Dropout(self.rnn_dropout)

    def _init_hidden(self, batch_size, device):
        """Initialize the GRU cell hidden state at start of training.
            User can choose from zero, random, xavier, kaiming-uniform

        Args:
            batch_size (int): Batch size to use during training.
            device (torch.device): Device to run on (CPU, CUDA, MPS).

        Returns: 
            h0 (torch.tensor): initialized torch tensor.
        
        Raises: 
            ValueError: Unsupported initialization string passed to function.
        """
        if self.hidden_init == 'zero':
            # Set all values to zero
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        elif self.hidden_init == 'rand':
            # Random initialization
            h0 = torch.randn(self.n_layers, batch_size, self.hidden_size, device=device)
        elif self.hidden_init == 'xavier':
            # Use xavier initialization
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            nn.init.xavier_uniform_(h0)
        elif self.hidden_init == 'he':
            h0 = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            nn.init.kaiming_uniform_(h0, mode='fan_in', nonlinearity='leaky_relu')
        else: raise ValueError(f"Unsupported hidden_init: {self.hidden_init}")

        return h0

    def forward(self, x):      
        """Forward pass of PyTorch model.

        Args:
          x: Input tensor to run through model.
        """
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
