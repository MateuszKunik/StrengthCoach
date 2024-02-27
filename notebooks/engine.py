import torch
from tqdm.auto import tqdm


def recursion(model, data):
    """
    
    """
    # Initialize the hidden state tensor depending on the actual batch size
    hidden_tensor = model.init_hidden_state(data.size(0))

    # Recursion procedure loop
    for i in range(data.size(1)):
        frames = data[:, i, :]
        hidden_tensor, outputs = model(frames, hidden_tensor)

    return outputs


def train_step(model, dataloader, loss_fn, optimizer, device):
    """

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
        predictions = recursion(model, data)

        # Calculate and accumulate loss
        loss = loss_fn(targets, predictions)
        accumulated_loss += loss.item()

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

    # Get average loss per batch
    accumulated_loss = accumulated_loss / len(dataloader)

    return accumulated_loss


def test_step(model, dataloader, loss_fn, device):
    """

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
            predictions = recursion(model, data)

            # Calculate and accumulate loss
            loss = loss_fn(targets, predictions)
            accumulated_loss += loss.item()

    # Get average loss per batch
    accumulated_loss = accumulated_loss / len(dataloader)

    return accumulated_loss


def train(model, train_dataloader, valid_dataloader, optimizer, loss_fn, n_epochs, device):
    """
    
    """
    # Prepare a train and validation loss storage
    results = {'train_loss': [], 'valid_loss': []}

    # Loop through training and validation steps for a number of epochs
    for epoch in tqdm(range(n_epochs)):
        train_loss = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
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

    return results