import numpy as np
import torch

def iou(pred, target, n_classes = 21):
    """
    Calculate the Intersection over Union (IoU) for predictions.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.
        n_classes (int, optional): Number of classes. Default is 21.

    Returns:
        float: Mean IoU across all classes.
    """
    # Convert boundary pixels (255) to background (0)
    target = torch.where(target == 255, torch.zeros_like(target), target)
    
    # Flatten the tensors
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Initialize arrays to store intersection and union for each class
    intersection = torch.zeros(n_classes)
    union = torch.zeros(n_classes)
    
    # Calculate IoU for each class
    for cls in range(n_classes):
        pred_cls = pred == cls
        target_cls = target == cls
        
        intersection[cls] = (pred_cls & target_cls).sum().float()
        union[cls] = (pred_cls | target_cls).sum().float()
    
    # Avoid division by zero
    valid = union > 0
    iou_per_class = torch.where(valid, intersection / union, torch.zeros_like(union))
    
    # Calculate mean IoU over valid classes
    mean_iou = iou_per_class[valid].mean().item()
    
    return mean_iou

def pixel_acc(pred, target):
    """
    Calculate pixel-wise accuracy between predictions and targets.

    Args:
        pred (tensor): Predicted output from the model.
        target (tensor): Ground truth labels.

    Returns:
        float: Pixel-wise accuracy.
    """
    # Convert boundary pixels (255) to background (0)
    target = torch.where(target == 255, torch.zeros_like(target), target)
    
    # Flatten the tensors
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Calculate accuracy
    correct = (pred == target).sum().float()
    total = target.numel()
    
    return (correct / total).item()