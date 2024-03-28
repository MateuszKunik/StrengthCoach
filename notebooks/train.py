import os
import numpy as np
import pandas as pd


import torch
from torch.nn import MSELoss

import engine
from model_builder import VanillaRNN
from utils import split_data, save_model
from data_setup import create_dataloaders


# Setup hyperparameters
BATCH_SIZE = 64
NUM_WORKERS = 0
PIN_MEMORY = True

INPUT_SIZE = 78
HIDDEN_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.float64

NUM_EPOCHS = 100
LEARNING_RATE = 0.001

# Setup directories
data_dir = '../data/processed/ProcessedSquats.csv'

# Read prepared dataframe
data = pd.read_csv(data_dir)

# Get file IDs splitted into train, validation and test 
file_ids = split_data(data)

# Create DataLoaders based on data_setup.py
train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
    data=data,
    file_ids=file_ids,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

# Create model based on model_builder.py
model = VanillaRNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    device=DEVICE,
    dtype=DTYPE
).to(DEVICE)

# Set loss and optimizer
loss_fn = MSELoss()
optimizer = torch.optim.Adam(
    params=model.parameters(),
    lr=LEARNING_RATE
)

# Start training using engine.py
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    optimizer=optimizer,
    loss_fn=loss_fn,
    n_epochs=NUM_EPOCHS,
    device=DEVICE
)

# Save the model with help from utils.py
# save_model()