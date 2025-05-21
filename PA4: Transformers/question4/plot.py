import matplotlib.pyplot as plt

def plot_accuracy(training_acc, validation_acc, save_path="accuracy_plot.png"):
    """
    Plots training and validation accuracy over epochs and saves the plot.
    
    :param training_acc: List of training accuracies per epoch.
    :param validation_acc: List of validation accuracies per epoch.
    :param save_path: Path to save the plot image.
    """
    epochs = range(1, len(training_acc) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, training_acc, 'b-', label='Training Accuracy')  # Blue line
    plt.plot(epochs, validation_acc, 'orange', label='Validation Accuracy')  # Orange line
    
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Warmup and LLRD Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.show()

# Example usage
training_acc = [0.369, 0.76, 0.892, 0.93, 0.947, 0.963, 0.9687, 0.9786, 0.9858, 0.99]
validation_acc = [0.66, 0.87, 0.8888, 0.8908, 0.8937, 0.8986, 0.9045, 0.9109, 0.9114, 0.9090]

plot_accuracy(training_acc, validation_acc, "./plots/training_vs_validation_accuracy.png")