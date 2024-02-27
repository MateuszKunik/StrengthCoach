import os
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
    

def create_dataloaders(data, file_ids, batch_size, num_workers, pin_memory):
    """
    
    """
    # Calculate max frequency needed for a padding
    max_frequency = data['MaxFrequency'] = data.groupby(by='FileId').size().max()
    
    # Get a dataframe, dataset, and dataloader for train file ids
    train_data = data.loc[
        data['FileId'].isin(file_ids["train"])]

    train_dataset = CustomDataset(
        train_data, max_frequency, transform=Norm())

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