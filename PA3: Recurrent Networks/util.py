import random
import torch
import matplotlib.pyplot as plt
import os
import numpy as np

def encode_text(input_file_path):
    # Load text data
    with open(input_file_path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Create character mapping
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    # Convert text to numerical format
    encoded_text = [char_to_idx[c] for c in text]

    return encoded_text, vocab_size, char_to_idx, idx_to_char

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length]) # creating a text sequence
        y.append(data[i+seq_length]) # the next character

    return np.array(X), np.array(y)


def plot_losses(train_losses, val_losses, fname):
    """
    Plots the training and validation losses across epochs and saves the plot as an image file with name - fname(function argument). 

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        fname (str): Name of the file to save the plot (without extension).

    Returns:
        None
    """

    # Create 'plots' directory if it doesn't exist

    if not os.path.isdir('plots'):
        os.mkdir('plots')

    # Plotting training and validation losses
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss per Epoch')

    # Saving the plot as an image file in 'plots' directory
    plt.savefig("./plots/" + fname + ".png")

