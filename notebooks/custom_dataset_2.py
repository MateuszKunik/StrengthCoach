import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.unique_files = data["FileId"].unique()

    def __len__(self):
        return len(self.unique_files)

    def __getitem__(self, idx):
        file_id = self.unique_files[idx]
        frames = torch.Tensor(
            self.data[self.data["FileId"] == file_id].values[:, 1:].astype(float)
        )
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
        indices = list(range(len(self.dataset)))
        indices.sort(key=lambda i: len(self.dataset[i][0]), reverse=True)
        batches = [
            indices[i : i + self.batch_size]
            for i in range(0, len(indices), self.batch_size)
        ]
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
        to_beginning = np.repeat([first_record], front, axis=0)
        to_end = np.repeat([last_record], back, axis=0)
        if len(data.shape) == 1:
            return np.hstack([to_beginning, data, to_end])
        elif len(data.shape) == 2:
            return np.vstack([to_beginning, data, to_end])
    elif difference == 1:
        last_record = data[-1]
        if len(data.shape) == 1:
            return np.hstack([data, last_record])
        elif len(data.shape) == 2:
            return np.vstack([data, last_record])
    else:
        return data


def collate_fn(batch):
    batch_frames = [item[0] for item in batch]
    batch_targets = [item[1] for item in batch]

    max_frames = max(len(frames) for frames in batch_frames)

    padded_frames = [add_padding(frames, max_frames) for frames in batch_frames]
    padded_targets = [add_padding(targets, max_frames) for targets in batch_targets]
    return padded_frames, padded_targets