import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    """
    
    """
    def __init__(self, data, transform=None):
        # Initialize 
        self.data = data
        self.transform = transform

        # Prepare tensor storage
        self.tensor = torch.tensor([])
        
        tmp = self.data.groupby(by='FileId', as_index=False).size()
        tmp = tmp.rename(columns={'size': 'Frequency'})
        self.data = pd.merge(self.data, tmp)

        self.data['MaxFrequency'] = self.data.groupby(by='FileId').size().max()

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
            # Concatenate to other tensors
            self.tensor = torch.cat((self.tensor, file_tensor), dim=0)
    

    def floor_ceil(self, x):
        """
        
        """
        return int(np.floor(x)), int(np.ceil(x))
    
    
    def add_padding(self, data):
        """

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
        
        """
        return self.tensor.shape[0]
    

    def __getitem__(self, index):
        """
        
        """
        # Get sample from data based on index
        sample = self.tensor[index, : :]

        # Extract features and target
        features = sample[:, :-1]
        target = sample[0, -1].unsqueeze(dim=0)

        # Transform if necessary
        if self.transform:
            return self.transform(features), target
        else:
            return features, target