import os
import torch
from tqdm.auto import tqdm
from datetime import datetime
from utils import save_checkpoints, plot_curves


def train_step(model, dataloader, loss_fn, optimizer, lr_scheduler=None, device='cuda'):
    """
    Performs a single training step for the model.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        loss_fn (callable): Loss function to use.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        lr_scheduler (optional): Learning rate scheduler.
        device (str, optional): Device to run the training on. Default is 'cuda'.

    Returns:
        float: Average loss per batch.
    """
    # Put model in train mode
    model.train()

    # Initialize accumulated loss
    accumulated_loss = 0.0

    # Loop through DataLoader batches
    for data, targets in dataloader:
        # Send data to target device
        data, targets = data.to(device), targets.to(device)

        # Forward pass
        predictions = model(data)

        # Calculate and accumulate loss
        loss = loss_fn(targets, predictions)
        accumulated_loss += loss.item()

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

    # Learning rate scheduler step
    if lr_scheduler is not None:
        lr_scheduler.step()

    # Get average loss per batch
    accumulated_loss = accumulated_loss / len(dataloader)

    return accumulated_loss


def test_step(model, dataloader, loss_fn, device='cuda'):
    """
    Performs a single evaluation step for the model.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        loss_fn (callable): Loss function to use.
        device (str, optional): Device to run the evaluation on. Default is 'cuda'.

    Returns:
        float: Average loss per batch.
    """
    # Put model in evaluation mode
    model.eval()

    # Initialize accumulated loss
    accumulated_loss = 0.0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for data, targets in dataloader:
            # Send data to target device
            data, targets = data.to(device), targets.to(device)

            # Forward pass
            predictions = model(data)

            # Calculate and accumulate loss
            loss = loss_fn(targets, predictions)
            accumulated_loss += loss.item()

    # Get average loss per batch
    accumulated_loss = accumulated_loss / len(dataloader)

    return accumulated_loss


def train(
        model, train_dataloader, valid_dataloader, loss_fn, optimizer,
        lr_scheduler=None, init_stopper=None, early_stopper=None,
        n_epochs=100, device='cuda', target_dir='../models'
):
    """
    Trains and validates the model over a specified number of epochs.

    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (torch.utils.data.DataLoader): DataLoader for training data.
        valid_dataloader (torch.utils.data.DataLoader): DataLoader for validation data.
        loss_fn (callable): Loss function to use.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        lr_scheduler (optional): Learning rate scheduler.
        init_stopper (optional): Initial stopper for early stopping.
        early_stopper (optional): Early stopping mechanism.
        n_epochs (int, optional): Number of epochs to train for. Default is 100.
        device (str, optional): Device to run the training on. Default is 'cuda'.
        target_dir (str, optional): Directory to save the model checkpoints. Default is '../models'.

    Returns:
        dict: Dictionary containing training and validation loss history.
    """
    # Get current time
    time = datetime.now().strftime('%d%m%y_%H%M')

    # Prepare a train and validation loss storage
    results = {'train_loss': [], 'valid_loss': []}

    # Loop through training and validation steps for a number of epochs
    for epoch in tqdm(range(n_epochs)):
        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            device=device
        )

        valid_loss = test_step(
            model=model,
            dataloader=valid_dataloader,
            loss_fn=loss_fn,
            device=device
        )

        # Print out what's happening
        print(f"Epoch: {epoch + 1} | train loss: {train_loss:.4f} | validation loss: {valid_loss:.4f}")

        results['train_loss'].append(train_loss)
        results['valid_loss'].append(valid_loss)


        if init_stopper.stop(valid_loss) or early_stopper.stop(valid_loss):
            break


    # Get model and optimizer state
    checkpoints = {
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': lr_scheduler
    }

    # Create a path to the model directory
    model_dir = os.path.join(target_dir, time)

    # Create a directory if necessary
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save model and optimizer state
    save_checkpoints(checkpoints, model_dir)
    # Plot loss curves
    plot_curves(results, model_dir)

    return results