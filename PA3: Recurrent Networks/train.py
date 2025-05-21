from util import *
from train import *
import torch.optim as optim
import torch.nn as nn
import os

from tqdm import tqdm

from shakespeare_lstm import LSTMModel
from shakespeare_rnn import RNNModel

def train(model, device, train_dataloader, val_dataloader, config):
    # AI prompt: same as the writeup PDF.
    """Train the model and save the best weights based on validation loss"""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        model.train()
        total_train_loss = 0
        
        # Training loop
        for batch_x, batch_y in tqdm(train_dataloader, desc=f'Epoch {epoch+1}'):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            output = model(batch_x)
            
            # Reshape output for loss calculation
            output = output.view(-1, output.size(-1))
            batch_y = batch_y.view(-1)
            
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation
        val_loss = eval(model, device, val_dataloader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}')
        
        # Save best model and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'models/{config["model"]}_seq_len_{config["seq_len"]}.pth')
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f'Early stopping after {epoch+1} epochs')
                # break
    
    # Plot and save the loss curves
    plot_losses(train_losses, val_losses, f'{config["model"]}_losses')
    
    return train_losses, val_losses

def eval(model, device, val_dataloader):
    # AI prompt: same as the writeup PDF.
    """Evaluate the model on validation/test data"""
    
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in val_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            output = model(batch_x)
            output = output.view(-1, output.size(-1))
            batch_y = batch_y.view(-1)
            
            loss = criterion(output, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(val_dataloader)
