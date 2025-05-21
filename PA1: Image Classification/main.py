import gradient
from constants import *
from train import *
from gradient import *
import argparse
import util 
from matplotlib import pyplot as plt

# TODO
def main(args):

    # Read the required config
    # Create different config files for different experiments
    configFile = None  # Will contain the name of the config file to be loaded
    if (args.experiment == 'test_softmax'):  # Rubric #4: Softmax Regression
        configFile = "config_4.yaml"
    elif (args.experiment == 'test_gradients'):  # Rubric #5: Numerical Approximation of Gradients
        configFile = "config_5.yaml"
    elif (args.experiment == 'test_momentum'):  # Rubric #6: Momentum Experiments
        configFile = "config_6.yaml"
    elif (args.experiment == 'test_regularization'):  # Rubric #7: Regularization Experiments
        configFile = "config_7.yaml"  # Create a config file and change None to the config file name
    elif (args.experiment == 'test_activation'):  # Rubric #8: Activation Experiments
        configFile = "config_8.yaml"  # Create a config file and change None to the config file name


    # Load the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(path=datasetDir)
    # Load the configuration from the corresponding yaml file. Specify the file path and name
    config = util.load_config(configYamlPath + configFile)
    
    # print("x_train.shape: ", x_train.shape)
    # print("y_train.shape: ", y_train.shape)
    # print("x_valid.shape: ", x_valid.shape)
    # print("y_valid.shape: ", y_valid.shape)
    # print("x_test.shape: ", x_test.shape)
    # print("y_test.shape: ", y_test.shape)
    
    
    # AI prompt: finish the main method
    # Create a neural network model with the configuration
    model = Neuralnetwork(config)

    if args.experiment == 'test_gradients':
        print("Running Gradient Validation...")

        x_train_small = x_train[:1]  # Single training example
        y_train_small = y_train[:1]  # Single target label

        # Perform gradient checking
        gradient_results = checkGradient(x_train_small, y_train_small, config)

        # Display results
        print("\nGradient Validation Results:")
        print(gradient_results)
        return


    # Train the model
    trained_model, trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, stop_epoch = train(model, x_train, y_train, x_valid, y_valid, config)
    
    # Test the model using modelTest function
    test_loss, test_accuracy = modelTest(trained_model, x_test, y_test)
    print(f"Test Loss from modelTest: {test_loss:.4f}")
    print(f"Test Accuracy from modelTest: {test_accuracy:.4f}")
    
    # AI prompt: For the code present, we get this error: "trainEpochLoss" is not defined. How can I resolve this? If you propose a fix, please make it concise.
    # plots(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop=None)
    util.plots(trainEpochLoss, trainEpochAccuracy, valEpochLoss, valEpochAccuracy, earlyStop=stop_epoch if config['early_stop'] else None)
    

if __name__ == "__main__":

    #Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='test_momentum', help='Specify the experiment that you want to run')
    args = parser.parse_args()
    main(args)