import torch


class Normalization(object):
    """
    Applies normalization to a tensor, scaling its values to the range [-1, 1].

    Methods:
        __call__(tensor): Normalizes the input tensor.
    """
    def __call__(self, tensor):
        """
        Normalizes the input tensor to the range [-1, 1].

        Args:
            tensor (torch.Tensor): Input tensor to be normalized.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Calculate minimal and maximal values contained in tensor
        min_value = tensor.min()
        max_value = tensor.max()

        # Normalization procedure
        normalized = 2 * (tensor - min_value) / (max_value - min_value) - 1

        return normalized
    

class AddGaussianNoise(object):
    """
    Adds Gaussian noise to a tensor with a specified probability.

    Attributes:
        p (float): Probability of adding noise.
        mean (float): Mean of the Gaussian noise.
        std (float): Standard deviation of the Gaussian noise.
    
    Methods:
        __call__(tensor): Adds Gaussian noise to the input tensor based on the specified probability.
    """
    def __init__(self, p=0.5, mean=0., std=1.):
        """
        Initializes the AddGaussianNoise with probability, mean, and standard deviation.

        Args:
            p (float): Probability of adding noise. Default is 0.5.
            mean (float): Mean of the Gaussian noise. Default is 0.
            std (float): Standard deviation of the Gaussian noise. Default is 1.
        """
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Adds Gaussian noise to the input tensor based on the specified probability.

        Args:
            tensor (torch.Tensor): Input tensor to add noise to.

        Returns:
            torch.Tensor: Tensor with added Gaussian noise if the probability condition is met; otherwise, the original tensor.
        """
        if self.p >= torch.rand(1):
            return tensor + torch.randn_like(tensor) * self.std + self.mean
        else:
            return tensor