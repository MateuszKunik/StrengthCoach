import torch

# Compute related
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 0

# Dataset setup and hyperparameters
DATA_DIR = '../data/processed/OneRepMaxData_010424.csv'
TARGET_DIR = '../models/squat/P1RM_Predictor'

# Dataset setup and hyperparameters
PROPORTIONS = [0.8, 0.15, 0.05]
BATCH_SIZE = 64
NUM_WORKERS = 0
PIN_MEMORY = True

# Training hyperparameters
INPUT_SIZE = 78
HIDDEN_SIZE = 256
NUM_LAYERS = 2

NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
STEP_SIZE = 20
GAMMA = 0.5