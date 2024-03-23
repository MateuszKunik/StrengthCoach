import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset


class CustomDataset(Dataset):
    # Initialize dataset with provided data
    def __init__(self, data):
        self.data = data
        # Get unique file IDs
        self.unique_files = data["FileId"].unique()

    def __len__(self):
        # Return the number of unique files in the dataset
        return len(self.unique_files)

    def __getitem__(self, idx):
        # Retrieve data corresponding to the given index
        file_id = self.unique_files[idx]
        # Extract frames for the given file ID and convert to Tensor
        frames = torch.Tensor(
            self.data[self.data["FileId"] == file_id].values[:, 1:].astype(float)
        )
        # Extract target values for the given file ID and convert to Tensor
        target_value = torch.Tensor(
            self.data[self.data["FileId"] == file_id]["PercentageMaxLoad"].values
        )
        return frames, target_value


class PaddedBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size):
        super().__init__(sampler=None, batch_size=batch_size, drop_last=False)
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        # Create batches of indices based on length of frames
        indices = list(range(len(self.dataset)))
        # Sort indices based on the length of frames (descending order)
        indices.sort(key=lambda i: len(self.dataset[i][0]), reverse=True)
        # Divide indices into batches based on batch_size
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
        # Sort batches by length of frames
        batches = sorted(
            batches, key=lambda x: len(self.dataset[x[0]][0]), reverse=True
        )
        yield from batches


def floor_ceil(x):
    return int(np.floor(x)), int(np.ceil(x))


def add_padding(data, max_frames):
    difference = max_frames - data.shape[0]
    if difference > 1:
        front, back = floor_ceil(difference / 2)
        first_record, last_record = data[0], data[-1]
        # Pad data at the beginning and end
        to_beginning = torch.tensor(np.repeat([first_record], front, axis=0))
        to_end = torch.tensor(np.repeat([last_record], back, axis=0))
        if len(data.shape) == 1:
            # Stack tensors horizontally
            return torch.hstack([to_beginning, data.clone().detach(), to_end])
        elif len(data.shape) == 2:
            return torch.vstack([to_beginning, data.clone().detach(), to_end])
    elif difference == 1:
        last_record = data[-1]
        if len(data.shape) == 1:
            # Stack tensors horizontally
            return torch.hstack([data.clone().detach(), last_record])
        elif len(data.shape) == 2:
            # Stack tensors vertically
            return torch.vstack([data.clone().detach(), last_record])
    else:
        return data.clone().detach()


def collate_fn(batch):
    batch_frames = [item[0] for item in batch]
    batch_targets = [item[1] for item in batch]
    # Find maximum number of frames in the batch
    max_frames = max(len(frames) for frames in batch_frames)
    # Add padding to frames and targets
    padded_frames = [add_padding(frames, max_frames) for frames in batch_frames]
    padded_targets = [add_padding(targets, max_frames) for targets in batch_targets]
    return padded_frames, padded_targets
