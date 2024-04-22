import torch


class Normalization(object):
    """
    
    """
    def __call__(self, tensor):
        # Calculate minimal and maximal values contained in tensor
        min_value = tensor.min()
        max_value = tensor.max()

        # Normalization procedure
        normalized = 2 * (tensor - min_value) / (max_value - min_value) - 1

        return normalized
    

class AddGaussianNoise(object):
    """ 
    
    """
    def __init__(self, p=0.5, mean=0., std=1.):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if self.p >= torch.rand(1):
            return tensor + torch.randn_like(tensor) * self.std + self.mean
        else:
            return tensor