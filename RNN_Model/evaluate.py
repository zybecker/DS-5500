import torch
from torch.nn import MSELoss, L1Loss
from torch.nn.utils import  clip_grad_norm_
from torcheval.metrics.functional import r2_score

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

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
        
        # Backwards pass and step
        loss.backward()

        # Sort of experimental, but clip gradients if desired (1.0)
        if grad_clip: clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Step
        optimizer.step()
        
        # Get loss
        epoch_loss += loss.item()*inputs.size(0)  # loss * batch size
    return epoch_loss / len(dataloader.dataset)  # average loss 


def evaluate_model(model, dataloader, device, loss_fn='MSE'):
    model.eval()
    total_loss = 0
    with torch.no_grad():  # Disable gradients for evaluation
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)  # Ensure test inputs are on the same device
            # Make prediction
            outputs = model(inputs)

            # Get loss - per user input
            if loss_fn == 'MSE':
                criterion = MSELoss()
                loss = criterion(outputs, targets)
            elif loss_fn == 'RMSE':
                criterion = MSELoss()
                loss = torch.sqrt(criterion(outputs, targets))
            elif loss_fn == 'MAE':
                criterion = L1Loss()
                loss = criterion(outputs, targets)
            else: raise ValueError

            total_loss += loss.item()*inputs.size(0)  # loss * batch size
    return total_loss / len(dataloader.dataset)  # average loss


def calculate_R2_score(model, dataloader, device=DEVICE):
    model.eval()
    # Lists to save all values when working in batches
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            all_outputs.append(outputs)
            all_targets.append(targets)

    # Concatenate all batch outputs and targets
    all_outputs = torch.cat(all_outputs)
    all_targets = torch.cat(all_targets)

    # Calculate R2 score on the entire dataset
    R2_score_value = r2_score(all_outputs, all_targets)
    return R2_score_value.item()
