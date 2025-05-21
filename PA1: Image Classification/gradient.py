import numpy as np
from neuralnet import Neuralnetwork
from copy import deepcopy


def check_grad(model, x_train, y_train, weights_to_check, epsilon=1e-3):

    """
    TODO
    Checks if gradients computed numerically are within O(epsilon**2)

    Args:
        model: The neural network model to check gradients for.
        x_train: Small subset of the original train dataset.
        y_train: Corresponding target labels of x_train.
        epsilon: Small constant for numerical approximation.

    Prints gradient difference of values calculated via numerical approximation and backprop implementation.
    """
    # raise NotImplementedError("check_grad not implemented in gradient.py")

    gradient_differences = []

    # Loop through specified weights
    for layer_idx, weight_idx in weights_to_check:
        # Deep copy the model to avoid altering the original during gradient approximation
        model_copy = deepcopy(model)
        layer = model_copy.layers[layer_idx]

        # Save the original weight
        original_weight = layer.w[weight_idx]

        # E(w + epsilon)
        layer.w[weight_idx] = original_weight + epsilon
        loss_plus, _ = model_copy.forward(x_train, y_train)

        # E(w - epsilon)
        layer.w[weight_idx] = original_weight - epsilon
        loss_minus, _ = model_copy.forward(x_train, y_train)

        # Restore the original weight
        layer.w[weight_idx] = original_weight

        # Numerical gradient
        numerical_gradient = (loss_plus - loss_minus) / (2 * epsilon)

        # Backpropagation gradient
        model.forward(x_train, y_train)  # Perform forward pass
        model.backward()                 # Perform backward pass
        backprop_gradient = model.layers[layer_idx].dw[weight_idx]
        backprop_gradient = -backprop_gradient

        # Compute absolute difference
        abs_diff = abs(numerical_gradient - backprop_gradient)

        # Store results
        gradient_differences.append({
            "Type of Weight": f"Layer {layer_idx}, Weight {weight_idx}",
            "Numerical Gradient": numerical_gradient,
            "Backpropagation Gradient": backprop_gradient,
            "Absolute Difference": abs_diff
        })

    return gradient_differences

# # AI prompt: save the results in a table. don't use pandas or other packages except numpy
def save_results_to_file(results, filename):
    """Save gradient validation results to a text file in a tabular format"""
    header = "{:<25} {:>20} {:>20} {:>20}".format(
        "Weight Type", "Numerical Gradient", "Backpropagation Gradient", "Absolute Difference"
    )
    
    with open(filename, 'w') as f:
        f.write(header + '\n')
        f.write('-' * 85 + '\n')
        
        for result in results:
            line = "{:<25} {:>20.2e} {:>20.2e} {:>20.2e}".format(
                result['Type of Weight'],
                float(result['Numerical Gradient']),
                float(result['Backpropagation Gradient']),
                float(result['Absolute Difference'])
            )
            f.write(line + '\n')

def checkGradient(x_train, y_train, config):
    model = Neuralnetwork(config)

    weights_to_check = [
        (1, (0, 0)),  # Output bias weight
        (0, (1, 1)),  # Hidden bias weight
        (1, (1, 0)),  # Hidden-to-output weight
        (1, (3, 2)),  # Hidden-to-output weight
        (0, (3, 2)),  # Input-to-hidden weight
        (0, (4, 2)),  # Input-to-hidden weight
    ]

    gradient_differences = check_grad(model, x_train[:1], y_train[:1], weights_to_check, 0.01)

    # Save results to a file
    output_file = "gradient_validation_results.txt"
    save_results_to_file(gradient_differences, output_file)
    
    # Print formatted table to console
    print("\nGradient Check Results:")
    print("-" * 85)
    print("{:<25} {:>20} {:>20} {:>20}".format(
        "Weight Type", "Numerical", "Backprop", "Difference"
    ))
    print("-" * 85)
    
    for result in gradient_differences:
        print("{:<25} {:>20.2e} {:>20.2e} {:>20.2e}".format(
            result['Type of Weight'],
            float(result['Numerical Gradient']),
            float(result['Backpropagation Gradient']),
            float(result['Absolute Difference'])
        ))
    print("-" * 85)

    return gradient_differences