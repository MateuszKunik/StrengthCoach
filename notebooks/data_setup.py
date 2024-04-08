from torch.utils.data import DataLoader
from custom_dataset import CustomDataset
from custom_transforms import Normalization


def create_dataloaders(data, file_ids, train_transform, augmentation, batch_size, num_workers, pin_memory):
    """
    
    """
    # Calculate max frequency needed for a padding
    max_frequency = data.groupby(by='FileId').size().max()
    
    # Get a dataframe, dataset, and dataloader for train file IDs
    train_data = data.loc[
        data['FileId'].isin(file_ids["train"])]

    train_dataset = CustomDataset(
        train_data, max_frequency, transform=train_transform, augmentation=augmentation)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    # Get a dataframe, dataset, and dataloader for validation file IDs
    valid_data = data.loc[
        data['FileId'].isin(file_ids["validation"])]
    
    valid_dataset = CustomDataset(
        valid_data, max_frequency, transform=Normalization())

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    # Check if test IDs were appointed
    if len(file_ids.keys()) == 3:
        # Get a dataframe, dataset, and dataloader for test file IDs
        test_data = data.loc[
            data['FileId'].isin(file_ids["test"])]
        
        test_dataset = CustomDataset(
            test_data, max_frequency, transform=Normalization())

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            shuffle=False
        )

        return train_dataloader, valid_dataloader, test_dataloader
    
    else:
        return train_dataloader, valid_dataloader