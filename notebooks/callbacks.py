class InitStopper:
    """
    A simple early stopping mechanism based on the initial validation loss.

    Args:
        patience (int): Number of epochs to wait after initial validation loss has been reached. Default is 1.
    
    Attributes:
        patience (int): Number of epochs to wait after initial validation loss has been reached.
        init_validation_loss (float or None): Initial validation loss.
        counter (int): Counter for tracking the number of epochs since initial validation loss.
    """
    def __init__(self, patience=1):
        self.patience = patience
        self.init_validation_loss = None
        self.counter = 0

    def stop(self, validation_loss):
        """
        Checks whether to stop the training based on the initial validation loss.

        Args:
            validation_loss (float): Current validation loss.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if self.init_validation_loss:
            if self.init_validation_loss == validation_loss:
                self.counter += 1
                if self.counter >= self.patience:
                    return True         
        else:
            self.init_validation_loss = validation_loss

        return False


class EarlyStopper:
    """
    A more sophisticated early stopping mechanism based on minimum validation loss and minimum delta.

    Args:
        patience (int): Number of epochs to wait after minimum validation loss has been reached. Default is 1.
        min_delta (float): Minimum change in validation loss to qualify as improvement. Default is 0.
    
    Attributes:
        patience (int): Number of epochs to wait after minimum validation loss has been reached.
        min_delta (float): Minimum change in validation loss to qualify as improvement.
        counter (int): Counter for tracking the number of epochs since minimum validation loss.
        min_validation_loss (float): Minimum validation loss encountered so far.
    """
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def stop(self, validation_loss):
        """
        Checks whether to stop the training based on the minimum validation loss and minimum delta.

        Args:
            validation_loss (float): Current validation loss.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False