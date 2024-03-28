import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from custom_dataset import CustomDataset


class Norm(object):
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
    

def create_dataloaders(data, file_ids, batch_size, num_workers, pin_memory):
    """
    
    """
    # Calculate max frequency needed for a padding
    max_frequency = data.groupby(by='FileId').size().max()
    
    # Get a dataframe, dataset, and dataloader for train file ids
    train_data = data.loc[
        data['FileId'].isin(file_ids["train"])]

    train_dataset = CustomDataset(
        train_data,
        max_frequency,
        transform=transforms.Compose(
            [
                Norm(),
                AddGaussianNoise(p=0.75, mean=0., std=0.05)
            ]
        )
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    # Get a dataframe, dataset, and dataloader for validation file ids
    valid_data = data.loc[
        data['FileId'].isin(file_ids["validation"])]
    
    valid_dataset = CustomDataset(
        valid_data, max_frequency, transform=Norm())

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    # Get a dataframe, dataset, and dataloader for test file ids
    test_data = data.loc[
        data['FileId'].isin(file_ids["test"])]
    
    test_dataset = CustomDataset(
        test_data, max_frequency, transform=Norm())

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_dataloader, valid_dataloader, test_dataloader