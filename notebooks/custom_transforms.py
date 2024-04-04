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
    def __init__(self, p=0.5, n_cols=57, mean=0., std=1.):
        self.p = p
        self.n_cols = n_cols
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        if self.p >= torch.rand(1):
            # Split the tensor for augmentation and for save
            tensor_1, tensor_2 = tensor.split(split_size=self.n_cols, dim=1)

            # Add Gaussian noise to the landmark coordinates data
            tensor_1 = tensor_1 + torch.randn_like(tensor_1) * self.std + self.mean
            # Concatenate noised and saved tensors
            tensor = torch.cat((tensor_1, tensor_2), dim=1)
        
        return tensor