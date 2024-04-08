import pandas as pd
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import OneCycleLR

import config
import engine
from loss import RMSELoss
from model_builder import RNN
from utils import split_data
from data_setup import create_dataloaders
from custom_transforms import Normalization, AddGaussianNoise
from callbacks import InitStopper, EarlyStopper

if config.SEED:
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)


# Read prepared dataframe
data = pd.read_csv(config.DATA_DIR)

# Get file IDs splitted into train, validation and test 
file_ids = split_data(data, config.PROPORTIONS, config.SEED)

# Create data augmentation and transformation object
transform = Normalization()
augmentation = AddGaussianNoise(
    p=config.NOISE_PROB,
    mean=config.NOISE_MEAN,
    std=config.NOISE_STD
)

# Create DataLoaders based on data_setup.py
train_dataloader, valid_dataloader, test_dataloader = create_dataloaders(
    data=data,
    file_ids=file_ids,
    train_transform=transform,
    augmentation=augmentation,
    batch_size=config.BATCH_SIZE,
    num_workers=config.NUM_WORKERS,
    pin_memory=config.PIN_MEMORY
)

# Create model based on model_builder.py
model = RNN(
    input_size=config.INPUT_SIZE,
    hidden_size=config.HIDDEN_SIZE,
    num_layers=config.NUM_LAYERS)

model.to(config.DEVICE)

# Set loss, optimizer and scheduler
loss_fn = RMSELoss()
optimizer = Adam(model.parameters(), lr=config.LEARNING_RATE)
lr_scheduler = OneCycleLR(
    optimizer,
    max_lr = 1e-3,
    steps_per_epoch = 100,
    epochs = 10,
    anneal_strategy = 'cos'
)

# lr_scheduler = StepLR(optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)

# Set callbacks
init_stopper = InitStopper(patience=10)
early_stopper = EarlyStopper(patience=50, min_delta=1)

# Start training using engine.py
engine.train(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    early_stopper=early_stopper,
    init_stopper=init_stopper,
    lr_scheduler=lr_scheduler,
    n_epochs=config.NUM_EPOCHS,
    device=config.DEVICE,
    target_dir=config.TARGET_DIR
)