import copy
from neuralnet import *

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """
    # AI prompt: Implement mini-batch SGD to train the model. Implement Early Stopping. Use config to set parameters for training like learning rate, momentum, etc. Algorithm 1 screenshot.
    # Initialize variables for early stopping
    best_valid_loss = float('inf')
    patience_counter = 0
    best_model = None
    
    # Get parameters from config
    num_epochs = config['epochs']
    batch_size = config['batch_size'] 
    early_stop = config['early_stop']
    early_stop_epoch = config['early_stop_epoch']
    momentum = config['momentum']
    patience = 5
    
    num_samples = x_train.shape[0]
    num_batches = num_samples // batch_size
    
    # Initialize tracking lists
    train_epoch_loss = []
    train_epoch_acc = []
    val_epoch_loss = []
    val_epoch_acc = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        # Shuffle training data
        indices = np.random.permutation(num_samples)
        x_train = x_train[indices]
        y_train = y_train[indices]
        
        # Mini-batch training
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = start_idx + batch_size
            
            batch_x = x_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            # Forward pass
            model.targets = batch_y
            loss, acc = model(batch_x, batch_y)
            epoch_losses.append(loss)
            
            # Backward pass
            model.backward()
        
        # Track training metrics
        train_epoch_loss.append(np.mean(epoch_losses))
        _, train_acc = model(x_train, y_train)
        train_epoch_acc.append(train_acc)
        
        # Evaluate and track validation metrics
        valid_loss, valid_acc = model(x_valid, y_valid)
        val_epoch_loss.append(valid_loss)
        val_epoch_acc.append(valid_acc)
        
        print(f"Epoch {epoch + 1}, Train Loss: {train_epoch_loss[-1]:.4f}, Train Acc: {train_epoch_acc[-1]:.4f}, Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}")
        
        # Only check early stopping after early_stop_epoch when early_stop is enabled
        if early_stop and epoch >= early_stop_epoch:
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                best_model = copy.deepcopy(model)
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                curr_stop_epoch = epoch
                print(f"Early stopping triggered at epoch {curr_stop_epoch + 1}")
                # Return metrics only up to current epoch and the stopping epoch
                return (best_model,
                        train_epoch_loss[:curr_stop_epoch + 1],
                        train_epoch_acc[:curr_stop_epoch + 1],
                        val_epoch_loss[:curr_stop_epoch + 1],
                        val_epoch_acc[:curr_stop_epoch + 1],
                        curr_stop_epoch)
    
    return (model if not best_model else best_model, 
            train_epoch_loss, 
            train_epoch_acc, 
            val_epoch_loss, 
            val_epoch_acc,
            num_epochs - 1)  # Return 0-based index for consistency


def modelTest(model, X_test, y_test):
    """
    TODO
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """
    
    # Forward pass through the model to get predictions and loss
    test_loss, test_accuracy = model(X_test, y_test)
    
    return test_loss, test_accuracy