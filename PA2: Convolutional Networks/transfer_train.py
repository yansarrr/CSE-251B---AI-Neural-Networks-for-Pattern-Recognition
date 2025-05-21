from basic_fcn import *
import time
from torch.utils.data import DataLoader
import torch
import gc
import voc
import torchvision.transforms as standard_transforms
import util
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from transfer_fcn import *
# num_workers = multiprocessing.cpu_count()
num_workers = 4

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data) #xavier not applicable for biases



#TODO Get class weights
def getClassWeights():
    # TODO for Q4.c || Caculate the weights for the classes
    raise NotImplementedError

# normalize using imagenet averages
mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) 
input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        standard_transforms.Normalize(*mean_std)
    ])

target_transform = MaskToTensor()

train_dataset = voc.VOC('train', transform=input_transform, target_transform=target_transform)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
# test_dataset = voc.VOC('test', transform=input_transform, target_transform=target_transform)

# [Corrected test dataset] Split validation dataset into validation and test sets
val_size = len(val_dataset) // 2
test_size = len(val_dataset) - val_size
val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [val_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size= 16, shuffle=True, num_workers=num_workers)
val_loader = DataLoader(dataset=val_dataset, batch_size= 16, shuffle=False, num_workers=num_workers)
test_loader = DataLoader(dataset=test_dataset, batch_size= 16, shuffle=False, num_workers=num_workers)


epochs = 30

n_class = 21

fcn_model = Transfer_FCN(n_class=n_class)
fcn_model.apply(init_weights)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

optimizer = torch.optim.Adam(fcn_model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss() 

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs , eta_min= 1e-8) # 4(a)

fcn_model = fcn_model.to(device)


def plot_losses(train_losses, val_losses, best_epoch, early_stop_epoch):
    """
    Plot training and validation losses.
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
    """
    plt.figure(figsize=(10, 6))
    
    epochs = range(len(train_losses))
    
    plt.plot(epochs, train_losses, label="Train Loss", marker="o")
    plt.plot(epochs, val_losses, label="Validation Loss", marker="o")
    
    plt.scatter(early_stop_epoch - 1, val_losses[early_stop_epoch - 1], 
                color='green', s=150, marker='X', label="Early Stop Epoch")
    
    plt.scatter(best_epoch - 1, val_losses[best_epoch - 1], 
                color='red', s=150, marker='o', label="Best Accuracy Model")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("5b Transfer Learning Loss vs. Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig('5b_loss_plot.png')
    plt.close()


def train():
    best_iou_score = 0.0
    patience = 5
    patience_counter = 0
    best_model_path = 'best_model.pth'
    
    # Lists to store losses
    train_losses = []
    val_losses = []

    best_epoch = 0
    early_stop_epoch = 0

    for epoch in range(epochs):
        ts = time.time()
        epoch_train_losses = []  # Store losses for current epoch
        
        for iter, (inputs, labels) in enumerate(train_loader):
            # AI prompt: the same text from the pdf. 
            optimizer.zero_grad()

            # Transfer inputs and labels to device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = fcn_model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
            
            if iter % 20 == 0:
                print("epoch{}, iter{}, loss: {}".format(epoch, iter, loss.item()))

        # Calculate average training loss for this epoch
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)

        # Validation phase
        val_loss, current_miou_score = val(epoch)
        val_losses.append(val_loss)

        print("Finish epoch {}, time elapsed {}, training loss: {:.4f}, validation loss: {:.4f}".format(
            epoch, time.time() - ts, avg_train_loss, val_loss))

        scheduler.step() #4(a)
        print(f"learning rate: {scheduler.get_last_lr()}")

        # Save best model and check for early stopping
        if current_miou_score > best_iou_score:
            best_iou_score = current_miou_score
            best_epoch = epoch
            # Save the best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': fcn_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou_score': best_iou_score,
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, best_model_path)
            print(f"Saved best model with IoU: {best_iou_score:.4f}")
        else:
            patience_counter += 1
            print(f"IoU didn't improve for {patience_counter} epochs")

            # print("No early stopping.")

        # Early stopping
        if early_stop_epoch == 0 and patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            print(f"Best IoU score: {best_iou_score:.4f}")
            early_stop_epoch = epoch

    # Plot losses
    plot_losses(train_losses, val_losses, best_epoch, early_stop_epoch)

    # Load best model for testing
    checkpoint = torch.load(best_model_path)
    fcn_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with IoU: {checkpoint['best_iou_score']:.4f}")

#TODO
def val(epoch):
    """
    Validate the deep learning model on a validation dataset.

    - Set model to evaluation mode.
    - Disable gradient calculations.
    - Iterate over validation data loader:
        - Perform forward pass to get outputs.
        - Compute loss and accumulate it.
        - Calculate and accumulate mean Intersection over Union (IoU) scores and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the epoch.
    - Switch model back to training mode.

    Args:
        epoch (int): The current epoch number.

    Returns:
        tuple: Mean IoU score and mean loss for this validation epoch.
    """
    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad(): # we don't need to calculate the gradient in the validation/testing
        # AI prompt: the same text from the pdf. 
        for iter, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            
            losses.append(loss.item())
            
            # Calculate IoU and accuracy
            pred = outputs.max(1)[1].cpu().numpy()
            gt = labels.cpu().numpy()
            
            # Convert numpy arrays to torch tensors before passing to iou and pixel_acc
            pred_tensor = torch.from_numpy(pred)
            gt_tensor = torch.from_numpy(gt)
            
            mean_iou_scores.append(util.iou(pred_tensor, gt_tensor))
            accuracy.append(util.pixel_acc(pred_tensor, gt_tensor))
            
        mean_loss = np.mean(losses)
        mean_iou = np.mean(mean_iou_scores)
        mean_acc = np.mean(accuracy)

        print(f"Loss at epoch: {epoch} is {mean_loss}")
        print(f"IoU at epoch: {epoch} is {mean_iou}")
        print(f"Pixel acc at epoch: {epoch} is {mean_acc}")
        
        fcn_model.train()
        
        return mean_loss, mean_iou

 #TODO
def modelTest():
    """
    Test the deep learning model using a test dataset.

    - Load the model with the best weights.
    - Set the model to evaluation mode.
    - Iterate over the test data loader:
        - Perform forward pass and compute loss.
        - Accumulate loss, IoU scores, and pixel accuracy.
    - Print average loss, IoU, and pixel accuracy for the test data.
    - Switch model back to training mode.

    Returns:
        None. Outputs average test metrics to the console.
    """

    fcn_model.eval()  # Put in eval mode (disables batchnorm/dropout) !
    
    # AI prompt: the same text from the pdf. 
    losses = []
    mean_iou_scores = []
    accuracy = []

    with torch.no_grad():  # we don't need to calculate the gradient in the validation/testing
        for iter, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            
            losses.append(loss.item())
            
            # Calculate IoU and accuracy
            pred = outputs.max(1)[1].cpu().numpy()
            gt = labels.cpu().numpy()
            
            # Convert numpy arrays to torch tensors before passing to iou and pixel_acc
            pred_tensor = torch.from_numpy(pred)
            gt_tensor = torch.from_numpy(gt)
            
            mean_iou_scores.append(util.iou(pred_tensor, gt_tensor))
            accuracy.append(util.pixel_acc(pred_tensor, gt_tensor))

    print(f"Test Loss: {np.mean(losses)}")
    print(f"Test IoU: {np.mean(mean_iou_scores)}")
    print(f"Test Pixel Accuracy: {np.mean(accuracy)}")

    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!


def exportModel(inputs):    
    """
    Export the output of the model for given inputs.

    - Set the model to evaluation mode.
    - Load the model with the best saved weights.
    - Perform a forward pass with the model to get output.
    - Switch model back to training mode.

    Args:
        inputs: Input data to the model.

    Returns:
        Output from the model for the given inputs.
    """

    fcn_model.eval() # Put in eval mode (disables batchnorm/dropout) !
    
    saved_model_path = "Fill Path To Best Model"
    # TODO Then Load your best model using saved_model_path

    # AI prompt: the same text from the pdf. 
    fcn_model.load_state_dict(torch.load(saved_model_path))
    
    inputs = inputs.to(device)
    
    output_image = fcn_model(inputs)
    
    fcn_model.train()  #TURNING THE TRAIN MODE BACK ON TO ENABLE BATCHNORM/DROPOUT!!
    
    return output_image


if __name__ == "__main__":

    val(0)  # show the accuracy before training
    train()
    modelTest()

    # housekeeping
    gc.collect()
    torch.cuda.empty_cache()
