import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    Custom dataset for handling time-series data with optional transformation and augmentation.

    Attributes:
        transform (callable, optional): Transformation function to be applied to the features.
        augmentation (callable, optional): Augmentation function to be applied to the features.
        data (pd.DataFrame): Dataframe containing the dataset.
        tensor (torch.Tensor): Tensor containing the processed data.
        n_cols (int): Number of columns containing coordinates.
    """
    def __init__(self, data, max_frequency, transform=None, augmentation=None):
        """
        Initializes the dataset with data, maximum frequency, and optional transformation and augmentation functions.

        Args:
            data (pd.DataFrame): Dataframe containing the dataset.
            max_frequency (int): Maximum frequency for padding.
            transform (callable, optional): Transformation function to be applied to the features.
            augmentation (callable, optional): Augmentation function to be applied to the features.
        """
        # Initialize
        self.transform = transform
        self.augmentation = augmentation
        
        if self.augmentation:
            self.data, self.n_cols = self.change_order(data)
        else:
            self.data = data


        # Prepare tensor storage
        self.tensor = torch.tensor([])
        
        tmp = self.data.groupby(by='FileId', as_index=False).size()
        tmp = tmp.rename(columns={'size': 'Frequency'})
        self.data = pd.merge(self.data, tmp)

        self.data['MaxFrequency'] = max_frequency

        for _, file_data in self.data.groupby(by='FileId'):
            # Drop the FileId column
            file_data = file_data.drop(columns='FileId')

            # Add padding to dataframe
            adjusted = self.add_padding(file_data)
            adjusted = adjusted.drop(columns=['Frequency', 'MaxFrequency'])
            
            # Convert the adjusted dataframe to a numpy array
            array = adjusted.to_numpy()

            # Convert numpy array to pytorch tensor 
            file_tensor = torch.from_numpy(array).unsqueeze(dim=0)
            file_tensor = file_tensor.to(torch.float32)
            # Concatenate to other tensors
            self.tensor = torch.cat((self.tensor, file_tensor), dim=0)
    

    def change_order(self, data):
        """
        Changes the order of columns, placing coordinate columns at the front.

        Args:
            data (pd.DataFrame): Dataframe containing the dataset.

        Returns:
            tuple: Modified dataframe and the number of coordinate columns.
        """
        # Extract columns containing coordinates
        coordinates = list(data.filter(regex='X$|Y$|Z$').columns)
        # Get how many columns contain coordinates
        n_columns = len(coordinates)

        # Change the order of columns
        data = data[
            coordinates + [col for col in data if col not in coordinates]
        ]

        return data, n_columns


    def floor_ceil(self, x):
        """
        Computes the floor and ceiling values of a number.

        Args:
            x (float): Input number.

        Returns:
            tuple: Floor and ceiling values of the input number.
        """
        return int(np.floor(x)), int(np.ceil(x))
    
    
    def add_padding(self, data):
        """
        Adds padding to the dataframe to match the maximum frequency.

        Args:
            data (pd.DataFrame): Dataframe containing the dataset.

        Returns:
            pd.DataFrame: Dataframe with added padding.
        """
        # Reset index
        data = data.reset_index(drop=True)

        # Calculate how much padding should be added
        difference = data.loc[0, 'MaxFrequency'] - data.loc[0, 'Frequency']

        if difference > 1:
            # Calculate how many padding should be added to the beginning and to the end
            front, back = self.floor_ceil(difference / 2)

            # Get the first and last record
            first_record, last_record = data.iloc[0], data.iloc[-1]

            # Prepare data frames
            to_beginning = pd.concat(front * [pd.DataFrame([first_record])])
            to_end = pd.concat(back * [pd.DataFrame([last_record])])

            # Return concatenated data frames
            return pd.concat([to_beginning, data, to_end], ignore_index=True)

        elif difference == 1:
            # Get only the last record
            last_record = data.iloc[-1]

            # Return concatenated data frames
            return pd.concat([data, pd.DataFrame([last_record])], ignore_index=True)

        else:
            return data


    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.tensor.shape[0]
    

    def __getitem__(self, index):
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple: Features and target of the sample.
        """
        # Get sample from data based on index
        sample = self.tensor[index, : :]

        # Extract features and target
        features = sample[:, :-1]
        target = sample[0, -1].unsqueeze(dim=0)
        
        
        # Transform if necessary
        if self.transform:
            features = self.transform(features)
                
        # Augment if necessary
        if self.augmentation:
            # Separate features into augmented and original
            to_augment, original = features.split(split_size=self.n_cols, dim=1)
            # Custom data augmentation
            augmented = self.augmentation(to_augment)
            
            features = torch.cat((augmented, original), dim=1)

            return features, target
        else:
            return features, target