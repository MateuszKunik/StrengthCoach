import torch
import torch.nn as nn
    

class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) model.

    This class implements a basic RNN model using PyTorch's nn.LSTM module.

    Args:
        input_size (int): Number of expected features in the input.
        hidden_size (int): Number of features in the hidden state.
        num_layers (int): Number of recurrent layers. Default is 1.
    """
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        # Initialize 
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Initialize RNN layer
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        # Initialize fully connected layer for output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.ReLU(),
        )

        # self.fc = nn.Linear(hidden_size, 1)


    def forward(self, input_tensor):
        """
        Forward pass method to compute the output of the RNN model.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        # Initialize the hidden state and cell state as zero tensors
        hidden_tensor = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size)
        hidden_tensor = hidden_tensor.to(input_tensor.device)

        cell_tensor = torch.zeros(self.num_layers, input_tensor.size(0), self.hidden_size)
        cell_tensor = cell_tensor.to(input_tensor.device)

        # Forward prop
        output_tensor, (hidden_tensor, cell_tensor) = self.rnn(input_tensor, (hidden_tensor, cell_tensor))
        # hidden_tensor = hidden_tensor.reshape(hidden_tensor.shape[1], -1)
        
        # output = self.fc(hidden_tensor)

        # output tensor flow
        output_tensor = output_tensor[:, -1, :]
        output = self.fc(output_tensor)
        # output_tensor = output_tensor.reshape(output_tensor.shape[0], -1)
        # output = self.fc(output_tensor)

        return output