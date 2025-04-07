import torch
from torch.nn import MSELoss, L1Loss
from torch.nn.utils import  clip_grad_norm_
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))


def custom_loss(y, y_hat, epsilon=1e-8, alpha=1.0):
    # Ensure y and y_hat are tensors of the same shape
    assert y.shape == y_hat.shape, "y and y_hat must have the same shape"
    
    # Mean Squared Error term
    mse_loss = torch.mean((y - y_hat) ** 2)
    # Log-Squared Error term
    log_loss = torch.mean((torch.log(y + epsilon) - torch.log(y_hat + epsilon)) ** 2)
    # Final loss MSE + logMSE
    loss = mse_loss + alpha * log_loss
    return loss


def train_model(model, optimizer, dataloader, device=DEVICE, grad_clip=False, loss_fn='MSE'):
    model.train()   
    epoch_loss = 0
    for inputs, targets in dataloader:
        # Send to the right device
        inputs, targets = inputs.to(device), targets.to(device)
        # Zero the gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(inputs)
        # Calculate and log loss
        if loss_fn == 'MSE':
            criterion = MSELoss()
            loss = criterion(outputs, targets)
        elif loss_fn == 'RMSE':
            criterion = MSELoss()
            loss = torch.sqrt(criterion(outputs, targets))
        elif loss_fn == 'MAE':
            criterion = L1Loss()
            loss = criterion(outputs, targets)
        elif loss_fn == 'custom':
            criterion = custom_loss
            loss = criterion(y_hat=outputs, y=targets, epsilon=1e-1, alpha=5)
        else: raise ValueError(f"Unsupported loss function: {loss_fn}")

        # Backwards pass and step
        loss.backward()

        # Sort of experimental, but clip gradients if desired (1.0)
        if grad_clip: clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        # Step
        optimizer.step()
        
        # Get loss
        epoch_loss += loss.item()*inputs.size(0)  # loss * batch size
    return epoch_loss / len(dataloader.dataset)  # average loss 


def get_predictions(model, dataloader, device):
    model.eval()
    all_targets, all_outputs = [], []
    with torch.no_grad():  # Disable gradients for evaluation
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # Ensure test inputs are on the same device
            # Make prediction
            outputs = model(inputs)
            
            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    return all_targets, all_outputs


def evaluate_model(model, data_loader, device=DEVICE, loss_fn='RMSE'):

    model.eval()
    total_loss = 0.0
    total_samples = 0 

    with torch.no_grad():
        for inputs, targets in data_loader:
            # Move the data to the correct device
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
            elif loss_fn == 'custom':
                criterion = custom_loss
                loss = criterion(y_hat=predictions, y=targets, epsilon=1e-1, alpha=5)
            else: raise ValueError(f"Unsupported loss function: {loss_fn}")

            # Accumulate the loss for this batch
            total_loss += loss.item() * inputs.size(0)  # Multiply by batch size to get total loss
            total_samples += inputs.size(0)

    # Calculate the average loss
    average_loss = total_loss / total_samples
    return average_loss


def calculate_R2_score(targets, predictions):
    targets = np.array(targets)
    predictions = np.array(predictions)
    R2_score_value = r2_score(targets, predictions)
    return R2_score_value



def best_hyperparameter_results(results, loss='MSE', n_best=6):
    # Get rankings of each model by RSME
    mlp_rsmes = np.array([results[i][1] for i in results])
    order = mlp_rsmes.argsort()
    # Then, pull out the top n models and print their information
    n_best = 6
    best_models = [(ind.item(), results[ind][1]) for ind in order[0:n_best]]
    # Print out index of model in results dictionary and it's parameters
    print(f'{n_best} best models (index and values):')
    for m in best_models:
        print(m)

    ### Plot n best results
    n_columns = 3
    n_rows = int(np.ceil(n_best / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4, n_rows * 3))
    axes = axes.flatten()

    for i, model in enumerate(order[0:n_best]):
        # model is the specific model, which is represented by an index in the results dict
        axes[i].set_title(f'Rank {i}: Model {order[i]}\n{results[model][2]}', size=8)
        axes[i].plot(results[model][0], c='b', label='Validation')
        axes[i].scatter(x=len(results[model][0])-1, y=results[model][1], c='r', s=20, label='Test Avg.')
        if i == 0: axes[i].legend(loc='upper right', fontsize='small', shadow=True)  # only show legend for first plot
    fig.text(0.001, 0.5, f'Loss ({loss})', ha='center', va='center', rotation='vertical', fontsize=13)
    fig.text(0.5, 0.001, 'Epochs', ha='center', va='center', fontsize=13)
    plt.suptitle(f"Best {n_best} Performing Models, Ranked by {loss}", fontsize=13)
    plt.tight_layout()

    return None

def visualize_all_hyperparameter_results(results, loss, num_epochs):
    n_figures = len(results)
    n_columns = 3
    n_rows = int(np.ceil(n_figures / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(n_columns * 4, n_rows * 3))
    axes = axes.flatten()

    for i in range(n_figures):
        axes[i].set_title(f'Model {i}\n{results[i][2]}', size=8)
        axes[i].plot(results[i][0], c='b')
        axes[i].scatter(x=num_epochs-1, y=results[i][1], c='b', s=20)
    for j in range(n_figures, len(axes)):
        axes[j].axis('off')
    plt.suptitle("All Evaluated Models")
    fig.text(0.001, 0.5, 'Loss ({loss})', ha='center', va='center', rotation='vertical', fontsize=13)
    fig.text(0.5, 0.001, 'Epochs', ha='center', va='center', fontsize=13)
    plt.tight_layout()
