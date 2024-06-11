import torch
import torch.nn as nn


class RMSELoss(nn.Module):
    """
    Root Mean Squared Error (RMSE) loss function.

    This loss function calculates the RMSE between predicted and target values.

    Args:
        eps (float, optional): Small value to avoid division by zero in the square root. Default is 1e-6.
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, y_hat, y):
        """
        Forward pass method to compute the RMSE loss.

        Args:
            y_hat (torch.Tensor): Predicted values.
            y (torch.Tensor): Target values.

        Returns:
            torch.Tensor: Computed RMSE loss.
        """
        loss = torch.sqrt(self.mse(y_hat, y) + self.eps)
        return loss