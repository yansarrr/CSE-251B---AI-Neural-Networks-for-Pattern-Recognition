#-----------------Training for FCN----------------------

import os
import time
import gc
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms

from basic_fcn import *
import voc
import util

output_folder = input("Enter a folder name to save outputs: ")
os.makedirs(output_folder, exist_ok=True)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.normal_(m.bias.data)

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
input_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])
target_transform = MaskToTensor()

train_dataset = voc.VOC('train', transform=input_transform, target_transform=target_transform)
val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
val_size = len(val_dataset) // 2
test_size = len(val_dataset) - val_size
val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [val_size, test_size])

train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4)

epochs = 30
n_class = 21
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

fcn_model = FCN(n_class=n_class)
fcn_model.apply(init_weights).to(device)

optimizer = torch.optim.Adam(fcn_model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

def train():
    train_losses, val_losses = [], []
    iou_scores, pixel_accs = [], []

    best_val_loss = float("inf")  # Initialize best validation loss

    for epoch in range(epochs):
        ts = time.time()
        epoch_train_losses = []

        fcn_model.train()
        for iter, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            epoch_train_losses.append(loss.item())

            if iter % 20 == 0:
                print(f"Epoch {epoch}, Iter {iter}, Loss: {loss.item():.4f}")

        # Compute and store metrics
        avg_train_loss = np.mean(epoch_train_losses)
        train_losses.append(avg_train_loss)
        val_loss, iou, pixel_acc = val()
        val_losses.append(val_loss)
        iou_scores.append(iou)
        pixel_accs.append(pixel_acc)

        print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, IoU={iou:.4f}, Pixel Acc={pixel_acc:.4f}")
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_path = f"{output_folder}/best_model.pth"
            torch.save({'epoch': epoch, 'model_state_dict': fcn_model.state_dict()}, model_path)
            print(f"New Best Model Saved at Epoch {epoch} with Val Loss: {val_loss:.4f}")

    plot_metrics(train_losses, val_losses, iou_scores, pixel_accs)

def val():
    fcn_model.eval()
    losses, iou_scores, accuracy = [], [], []

    with torch.no_grad():
        for _, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = fcn_model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            pred = outputs.max(1)[1].cpu().numpy()
            gt = labels.cpu().numpy()

            pred_tensor, gt_tensor = torch.from_numpy(pred), torch.from_numpy(gt)
            iou_scores.append(util.iou(pred_tensor, gt_tensor))
            accuracy.append(util.pixel_acc(pred_tensor, gt_tensor))

    return np.mean(losses), np.mean(iou_scores), np.mean(accuracy)

def plot_metrics(train_losses, val_losses, iou_scores, pixel_accs):
    """ Generate and save plots for Loss, IoU, and Pixel Accuracy. """

    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss', marker='o')
    plt.plot(val_losses, label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_folder}/loss_plot.png")
    plt.close()

    # IoU Score Plot
    plt.figure(figsize=(10, 5))
    plt.plot(iou_scores, label='IoU Score', marker='o', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Mean IoU')
    plt.title('IoU Score Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_folder}/iou_plot.png")
    plt.close()

    # Pixel Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(pixel_accs, label='Pixel Accuracy', marker='o', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Pixel Accuracy')
    plt.title('Pixel Accuracy Over Epochs')
    plt.legend()
    plt.grid()
    plt.savefig(f"{output_folder}/pixel_acc_plot.png")
    plt.close()

if __name__ == "__main__":
    val()
    train()
    gc.collect()
    torch.cuda.empty_cache()

# #-----------------Training for UNet----------------------

# import torch
# import torch.nn as nn  
# import torch.optim as optim
# import os
# import time
# import gc
# import numpy as np
# import matplotlib.pyplot as plt
# from torch.utils.data import DataLoader
# import torchvision.transforms as standard_transforms

# from UNet import FCN_Unet  # Import U-Net model
# import voc
# import util

# output_folder = input("Enter a folder name to save outputs: ")
# os.makedirs(output_folder, exist_ok=True)

# def init_weights(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
#         torch.nn.init.xavier_uniform_(m.weight.data)
#         if m.bias is not None:
#             torch.nn.init.normal_(m.bias.data)

# class MaskToTensor(object):
#     def __call__(self, img):
#         return torch.from_numpy(np.array(img, dtype=np.int32)).long()

# mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# input_transform = standard_transforms.Compose([
#     standard_transforms.ToTensor(),
#     standard_transforms.Normalize(*mean_std)
# ])
# target_transform = MaskToTensor()

# train_dataset = voc.VOC('train', transform=input_transform, target_transform=target_transform)
# val_dataset = voc.VOC('val', transform=input_transform, target_transform=target_transform)
# val_size = len(val_dataset) // 2
# test_size = len(val_dataset) - val_size
# val_dataset, test_dataset = torch.utils.data.random_split(val_dataset, [val_size, test_size])

# train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True, num_workers=4)
# val_loader = DataLoader(dataset=val_dataset, batch_size=16, shuffle=False, num_workers=4)
# test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False, num_workers=4)

# epochs = 30
# n_class = 21
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fcn_model = FCN_Unet(n_class=n_class).to(device)
# fcn_model.apply(init_weights)

# optimizer = torch.optim.Adam(fcn_model.parameters(), lr=0.0005)
# criterion = torch.nn.CrossEntropyLoss()
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)

# patience = 5
# patience_counter = 0
# best_iou_score = 0.0
# best_epoch = 0
# early_stop_epoch = None
# best_model_path = f"{output_folder}/best_model.pth"

# def train():
#     train_losses, val_losses = [], []
#     iou_scores, pixel_accs = [], []

#     global best_iou_score, best_epoch, early_stop_epoch, patience_counter

#     for epoch in range(epochs):
#         ts = time.time()
#         epoch_train_losses = []

#         fcn_model.train()
#         for iter, (inputs, labels) in enumerate(train_loader):
#             optimizer.zero_grad()
#             inputs, labels = inputs.to(device), labels.to(device)

#             outputs = fcn_model(inputs)
#             loss = criterion(outputs, labels)

#             loss.backward()
#             optimizer.step()

#             epoch_train_losses.append(loss.item())

#             if iter % 20 == 0:
#                 print(f"Epoch {epoch}, Iter {iter}, Loss: {loss.item():.4f}")

#         # Compute and store metrics
#         avg_train_loss = np.mean(epoch_train_losses)
#         train_losses.append(avg_train_loss)
#         val_loss, iou, pixel_acc = val()
#         val_losses.append(val_loss)
#         iou_scores.append(iou)
#         pixel_accs.append(pixel_acc)

#         print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={val_loss:.4f}, IoU={iou:.4f}, Pixel Acc={pixel_acc:.4f}")
#         scheduler.step()

#         if iou > best_iou_score:
#             best_iou_score = iou
#             best_epoch = epoch
#             patience_counter = 0  # Reset patience counter
#             torch.save({'epoch': epoch, 'model_state_dict': fcn_model.state_dict()}, best_model_path)
#             print(f"New Best Model Saved at Epoch {epoch} with IoU: {iou:.4f}")
#         else:
#             patience_counter += 1
#             print(f"IoU didn't improve for {patience_counter} epochs.")

#         if patience_counter >= patience:
#             print(f"Early stopping triggered after {epoch + 1} epochs. Best IoU: {best_iou_score:.4f}")
#             early_stop_epoch = epoch
#             break  

#     plot_metrics(train_losses, val_losses, iou_scores, pixel_accs, best_epoch, early_stop_epoch)

# def val():
#     fcn_model.eval()
#     losses, iou_scores, accuracy = [], [], []

#     with torch.no_grad():
#         for _, (inputs, labels) in enumerate(val_loader):
#             inputs, labels = inputs.to(device), labels.to(device)

#             outputs = fcn_model(inputs)
#             loss = criterion(outputs, labels)
#             losses.append(loss.item())

#             pred = outputs.max(1)[1].cpu().numpy()
#             gt = labels.cpu().numpy()

#             pred_tensor, gt_tensor = torch.from_numpy(pred), torch.from_numpy(gt)
#             iou_scores.append(util.iou(pred_tensor, gt_tensor))
#             accuracy.append(util.pixel_acc(pred_tensor, gt_tensor))

#     return np.mean(losses), np.mean(iou_scores), np.mean(accuracy)

# def plot_metrics(train_losses, val_losses, iou_scores, pixel_accs, best_epoch, early_stop_epoch):
#     plt.figure(figsize=(10, 5))
#     plt.plot(train_losses, label='Training Loss', marker='o')
#     plt.plot(val_losses, label='Validation Loss', marker='o')

#     if early_stop_epoch is not None:
#         plt.scatter(early_stop_epoch, val_losses[early_stop_epoch], color='green', s=150, marker='X', label="Early Stop Epoch")

#     plt.scatter(best_epoch, val_losses[best_epoch], color='red', s=150, marker='o', label="Best IoU Model")

#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
#     plt.legend()
#     plt.grid()
#     plt.savefig(f"{output_folder}/loss_plot.png")
#     plt.close()

#     # IoU Score Plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(iou_scores, label='IoU Score', marker='o', color='green')
#     plt.xlabel('Epoch')
#     plt.ylabel('Mean IoU')
#     plt.title('IoU Score Over Epochs')
#     plt.legend()
#     plt.grid()
#     plt.savefig(f"{output_folder}/iou_plot.png")
#     plt.close()

#     # Pixel Accuracy Plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(pixel_accs, label='Pixel Accuracy', marker='o', color='red')
#     plt.xlabel('Epoch')
#     plt.ylabel('Pixel Accuracy')
#     plt.title('Pixel Accuracy Over Epochs')
#     plt.legend()
#     plt.grid()
#     plt.savefig(f"{output_folder}/pixel_acc_plot.png")
#     plt.close()

# if __name__ == "__main__":
#     val()
#     train()
#     gc.collect()
#     torch.cuda.empty_cache()
