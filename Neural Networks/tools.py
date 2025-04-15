"""
Functions to support tuning, training, and evaluation of neural network models.
Last updated: 4/11/2025
"""

import torch
from torch.nn import MSELoss, L1Loss
from torch.nn.utils import  clip_grad_norm_
from torch.utils.data import random_split
from torch.utils.data import TensorDataset
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Global setting for training device. User can still pass in perferred device during function calls
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def msle_loss(predictions, targets):
    return torch.mean((torch.log1p(predictions) - torch.log1p(targets)) ** 2)


def train_model(model, optimizer, dataloader, device=DEVICE, grad_clip=False, loss_fn='MSE'):
    """Script to train one epoch of a PyTorch model. 

    Args:
        model (nn.Module): PyTorch neural network model.
        optimizer (torch.optim): PyTorch optimizer.
        dataloader (DataLoader): PyTorch dataloader with inputs and targets.
        device (torch.device): Device to run on (CPU, CUDA, MPS).
        grad_clip (False or str): Value to clip gradients on (False if no clip).
        loss_fn (str): string declaring loss function to use.

    Returns:
        Average loss value over training epoch.

    Raises:
        ValueError: Unsupported loss function string passed to function.
    """
    model.to(device)  # Send to right device
    model.train()
    epoch_loss = 0  # Track loss over epoch
    # Go through all values in dataloader
    for inputs, targets in dataloader:
        # Send to the right device
        inputs, targets = inputs.to(device), targets.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Calculate and record loss
        if loss_fn == 'MSE':
            criterion = MSELoss()
            loss = criterion(outputs, targets)
        elif loss_fn == 'RMSE':
            criterion = MSELoss()
            loss = torch.sqrt(criterion(outputs, targets))
        elif loss_fn == 'MAE':
            criterion = L1Loss()
            loss = criterion(outputs, targets)
        elif loss_fn == 'MSLE':
            loss = msle_loss(outputs, targets)
        # If unsupported string passed in as loss function
        else: raise ValueError(f"Unsupported loss function: {loss_fn}")

        # Backwards pass and step
        loss.backward()
        # Clip gradients if desired
        if grad_clip and ~np.isnan(grad_clip): clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        # Step
        optimizer.step()
        
        # Get loss
        epoch_loss += loss.item()*inputs.size(0)  # loss * batch size
    return epoch_loss / len(dataloader.dataset)  # average loss over epoch


def get_predictions(model, dataloader, device=DEVICE):
    """Given a PyTorch model and dataloader, make predictions using model.

    Args:
        model (nn.Module): PyTorch neural network model.
        dataloader (DataLoader): PyTorch dataloader with inputs and targets.
        device (torch.device): Device to run on (CPU, CUDA, MPS).

    Returns:
        all_targets, all_outputs: lists of true targets and predicted values.
    """
    model.to(device)  # Send to right device
    model.eval()
    all_targets, all_outputs = [], []  # Lists to hold results
    with torch.no_grad():  # Disable gradients for evaluation
        # Go through all values in dataloader
        for inputs, targets in dataloader:
            # Send to the right device
            inputs, targets = inputs.to(device), targets.to(device)  
            # Make prediction
            outputs = model(inputs)
            # Append true and predicted values to list
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    return all_targets, all_outputs


def evaluate_model(model, dataloader, device=DEVICE, loss_fn='RMSE'):
    """Evaluate model predictions and true values using loss function.

    Args:
        model (nn.Module): PyTorch neural network model.
        dataloader (DataLoader): PyTorch dataloader with inputs and targets.
        device (torch.device): Device to run on (CPU, CUDA, MPS).
        loss_fn (str): string declaring loss function to use.

    Returns:
        average_loss (float): Average loss over all values in dataloader.

    Raises:
        ValueError: Unsupported loss function string passed to function.
    """
    model.to(device)  # Send to right device
    model.eval()
    # Tracking loss values and number of samples
    total_loss = 0.0
    total_samples = 0 

    with torch.no_grad():
        # Go through all values
        for inputs, targets in dataloader:
            # Send to the right device
            inputs, targets = inputs.to(device), targets.to(device)
            # Get model predictions
            predictions = model(inputs)

            # Choose the loss function
            if loss_fn == 'MSE':
                criterion = MSELoss()
                loss = criterion(predictions, targets)
            elif loss_fn == 'RMSE':
                criterion = MSELoss()
                loss = torch.sqrt(criterion(predictions, targets))
            elif loss_fn == 'MAE':
                criterion = L1Loss()
                loss = criterion(predictions, targets)
            elif loss_fn == 'MSLE':
                loss = msle_loss(predictions, targets)
            # If unsupported string passed in as loss function
            else: raise ValueError(f"Unsupported loss function: {loss_fn}")

            # Accumulate the loss for this batch
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size to get total loss
            total_samples += inputs.size(0)  # Track number of samples processed

    # Calculate the average loss over all values
    average_loss = total_loss / total_samples
    return average_loss


def calculate_R2_score(targets, predictions):
    """Calculate coefficient of determination (R2).

    Args:
        targets (list): true values.
        predictions (list): model-predicted values.

    Returns:
        R2_score_value (float): calculated R2 score.
    """
    # Convert input lists to numpy arrays
    targets = np.array(targets)
    predictions = np.array(predictions)
    # Calculate and return score using sklearn's 
    return r2_score(targets, predictions)


def best_hyperparameter_results(results, loss='MSE', n_best=6):
    """Process all hyperparameter tuning results and choose best

    Given a list dicts of results, will choose the n best results and
    print the best results and plot the training loss.

    Each list in the results dictionary will have (in order of index):
        List of loss value at each training epoch.
        Test accuracy value.
        Batch size.
        Number of training epochs (varies due to early stopping).


    Args:
        results (dict): Dictionary of lists of model training results.
            Key corresponds to model number training (sequentially).
        loss (str): String of loss function used to include in plot title.
        n_best (6): How many best models to print and plot.

    Returns:
        None. Prints results and visualizes all results in matplotlib plot.
    """
    # Get rankings of each model by loss
    mlp_rsmes = np.array([results[i][1] for i in results])
    order = mlp_rsmes.argsort()
    # Then, pull out the top n models and print their information
    n_best = 6
    best_models = [(ind.item(), results[ind][1]) for ind in order[0:n_best]]
    # Print out index of model in results dictionary and its parameters
    print(f'{n_best} best models (index and values):')
    for m in best_models:
        print(m)

    # Plot n best results
    n_columns = 3
    n_rows = int(np.ceil(n_best / n_columns))  # dynamically adjust row number

    fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4, n_rows * 3))
    axes = axes.flatten()

    for i, model in enumerate(order[0:n_best]):
        # Model is the specific model, which is represented by an index in the results dict
        axes[i].set_title(f'Rank {i}: Model {order[i]}\n{results[model][2]}', size=8)
        # Plot loss over training epochs and at end of training
        axes[i].plot(results[model][0], c='b', label='Validation Loss')
        axes[i].scatter(x=len(results[model][0])-1, y=results[model][1], c='r', s=20, label='Final Loss')
        if i == 0: axes[i].legend(loc='upper right', fontsize='small', shadow=True)  # only show legend for first plot
    fig.text(0.001, 0.5, f'Loss ({loss})', ha='center', va='center', rotation='vertical', fontsize=13)
    fig.text(0.5, 0.001, 'Epochs', ha='center', va='center', fontsize=13)
    plt.suptitle(f"Best {n_best} Performing Models, Ranked by {loss}", fontsize=13)
    plt.tight_layout()
    plt.show()

    return None

def visualize_all_hyperparameter_results(results, loss='MSE'):
    """If desired, visualize the results of all hyperparameter tunings.

    Will plot all results, which have hyperparameter and loss values
    in plot title. Might be overwhelming large, so use with caution.

    Each list in the results dictionary will have (in order of index):
        [0]: List of loss value at each training epoch.
        [1]: Test accuracy value.
        [2]: Batch size.
        [3]: Number of training epochs (varies due to early stopping).

    Args:
        results (dict): Dictionary of lists of model training results.
            Key corresponds to model number training (sequentially).
        loss (str): String of loss function used to include in plot title.

    Returns:
        None. Visualizes all results in matplotlib plot.
    """
    n_figures = len(results)
    n_columns = 3
    n_rows = int(np.ceil(n_figures / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4, n_rows * 3))
    axes = axes.flatten()

    for i in range(n_figures):  # In this case, i is also model #
        axes[i].set_title(f'Model {i}\n{results[i][2]}', size=8)
        # Plot loss over training epochs and at end of training
        axes[i].plot(results[i][0], c='b')
        axes[i].scatter(x=len(results[i][0])-1, y=results[i][1], c='b', s=20)
    for j in range(n_figures, len(axes)):
        axes[j].axis('off')
    plt.suptitle("All Evaluated Models")
    fig.text(0.001, 0.5, f'Loss ({loss})', ha='center', va='center', rotation='vertical', fontsize=13)
    fig.text(0.5, 0.001, 'Epochs', ha='center', va='center', fontsize=13)
    plt.tight_layout()
    plt.show()

    return None


class EarlyStopping:
    """Class to implement early stopping during model training.

    Class will track loss over training epoch. Upon enough epochs (patience)
    of no improvement (min_delta) in loss function, will communicate 
    to training script to terminate training.

    Attributes:
        patience: How many epochs training can not improve before terminating training.
        min_delta: Difference between best and current to flag counter.
        counter (int): How many epochs training has not improved.
        best_loss (float): Current best/lowest value for loss function.
        early_stop (boolean): Whether to stop early (True) or keep training (False).
    """
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        """Implementation of early stopping during model training.

        Args:
          val_loss (float): Numeric value of epoch loss.
        
        Returns: None.
        """
        # First run
        if self.best_loss is None:
            self.best_loss = val_loss
        # Improving - reset counter
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        # Not improving, but still not at criteria to stop
        elif self.counter < self.patience:
            self.counter += 1
        # Threshold met. Stop training
        else:
            self.early_stop = True


def get_dataloaders(dataset):

    # Split into test and train data
    train_dataset, test_dataset = random_split(dataset, [0.9, 0.1])
    train_dataset, valid_dataset = random_split(train_dataset, [0.89, 0.11])

    print(f"Train: {len(train_dataset)/len(dataset)*100:.1f}%, Test: {len(test_dataset)/len(dataset)*100:.1f}%, Valid: {len(valid_dataset)/len(dataset)*100:.1f}%")


    # Save as one big list
    datasets = [train_dataset, valid_dataset, test_dataset]

    # Convert datasets into tensors for easier manipulation
    inputs = [torch.stack([x[0] for x in set]) for set in datasets]

    # Store original shapes of each dataset
    original_shapes = [input.shape for input in inputs]

    # Reshape the inputs to 2D for scaling
    inputs_reshaped = [input.view(input.size(0), -1) for input in inputs]

    # Create scaler
    scaler = StandardScaler()

    # Fit scaler on training data and transform (use train scaler for all)
    inputs_scaled = [torch.tensor(scaler.fit_transform(input.numpy()), dtype=torch.float32) for input in inputs_reshaped]

    # Reshape the scaled inputs back to the original shape
    inputs_scaled_reshaped = [input.view(original_shapes[i]) for i, input in enumerate(inputs_scaled)]

    # Recreate the datasets with the scaled inputs
    train_dataset, valid_dataset, test_dataset = [TensorDataset(input, torch.stack([x[1] for x in datasets[i]])) for i, input in enumerate(inputs_scaled_reshaped)]

    # Print output shapes
    print(f"Train inputs shape: {inputs_scaled_reshaped[0].shape}. Mean, Std: [{inputs_scaled_reshaped[0].mean():.1f},{inputs_scaled_reshaped[0].std():.1f}]")
    print(f"Valid inputs shape: {inputs_scaled_reshaped[1].shape}. Mean, Std: [{inputs_scaled_reshaped[1].mean():.1f},{inputs_scaled_reshaped[1].std():.1f}]")
    print(f"Test inputs shape: {inputs_scaled_reshaped[2].shape}. Mean, Std: [{inputs_scaled_reshaped[2].mean():.1f},{inputs_scaled_reshaped[2].std():.1f}]")

    return train_dataset, valid_dataset, test_dataset

