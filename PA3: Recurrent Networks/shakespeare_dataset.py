import torch
from torch.utils.data import Dataset

# AI prompt: same as the writeup PDF.
class ShakespeareDataset(Dataset):
    def __init__(self, X, y):
        """
        Initialize the dataset with sequences and their corresponding targets
        
        Args:
            X (torch.Tensor): Input sequences
            y (torch.Tensor): Target characters
        """
        self.X = X
        self.y = y
        
    def __len__(self):
        """Return the total number of sequences"""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Return a single sequence and its target"""
        return self.X[idx], self.y[idx]
