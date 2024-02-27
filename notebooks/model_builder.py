import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """

    """
    def __init__(self, input_size, hidden_size, device='cuda', dtype=torch.float64):
        super(VanillaRNN, self).__init__()
        # Initialize 
        self.hidden_size = hidden_size
        self.device = device

        # Initialize input to hidden state layer
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size, dtype=dtype)
        # Initialize hidden state to output layer
        self.fc = nn.Linear(hidden_size, 1, dtype=dtype)


    def init_hidden_state(self, batch_size):
        """
        
        """
        return torch.zeros((batch_size, self.hidden_size), device=self.device)


    def forward(self, input_tensor, hidden_tensor):
        """
        
        """
        # Concatenate input and hidden tensors
        combined = torch.cat((input_tensor, hidden_tensor), dim=1)

        hidden = self.i2h(combined)
        output = self.fc(hidden)

        return hidden, output