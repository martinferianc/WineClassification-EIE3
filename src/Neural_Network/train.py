import os
import sys
import copy
import numpy as np

from network import Network
import tensorflow as tf
from postprocess import plot_confusion_matrix, calculate_scores

# Initialize the basic parameters of the network
LEARNING_RATE = 0.001
HIDDEN_LAYERS = 3
HIDDEN_NEURONS = 1
REGULARIZER = "L2"
REGULARIZATION_PENALTY = 0.001
ACTIVATION = "relu"
EPOCHS = 1
N_BATCHES = 10
DROPOUT = 0.8
BETA1= 0.99
TEST = False
BASE_NAME = str(HIDDEN_LAYERS) + "_" + str(HIDDEN_NEURONS) + "_" + str(DROPOUT) + "_" + str(REGULARIZER) + "_" + str(REGULARIZATION_PENALTY) + "_" + str(ACTIVATION)

# Convert labels into one-hot encoding
def one_hot(labels):
    n_values = 11
    labels.astype(int)
    labels = np.eye(n_values)[labels]
    return labels

# Separete the labels from the features and convert the labels into one-hot representation
def separete_data(data):
    features = np.array(copy.deepcopy([x[:-1] for x in data]), dtype=np.float64)
    labels = np.array(copy.deepcopy([x[-1] for x in data]), dtype=np.int32)
    original_labels = copy.deepcopy(labels)
    one_hot_labels = one_hot(labels)
    return (features, one_hot_labels, original_labels)

# This function performs training for the specific network and performs n_fold cross validation
def n_fold(n_folds = 10, save = True, test = False):
    accuracies = []

    print("### Beginning n-fold cross validation with parameters: ###")
    print("## Netwok name: ##".format(BASE_NAME))
    print("## Learning rate:{} ##".format(LEARNING_RATE))
    print("## Hidden neurons:{} ##".format(HIDDEN_NEURONS))
    print("## Hidden layers:{} ##".format(HIDDEN_LAYERS))
    print("## Epochs:{} ##".format(EPOCHS))
    print("## Batch Size:{} ##".format(N_BATCHES))
    print("## Regularizer:{} ##".format(REGULARIZER))
    print("## Regularizer penalty:{} ##".format(REGULARIZATION_PENALTY))
    print("## Activation:{} ##".format(ACTIVATION))

    if not os.path.isdir("models/"+BASE_NAME):
        os.makedirs("models/"+BASE_NAME)

    for i in range(0,n_folds):
        print("## Fold:{} ##".format(i))
        # Begin each training completely separetely

        with tf.Graph().as_default():
            # Load all the data into the work place
            train_data = np.load(os.path.join("..", "data", "processed", "{}_training.npy".format(i)))
            validation_data = np.load(os.path.join("..", "data", "processed", "{}_validation.npy".format(i)))
            test_data = np.load(os.path.join("..", "data", "processed",  "{}_test.npy".format(i)))
            if test:
                # Join training and validation data for the final test
                train_data = np.concatenate((train_data, validation_data), axis=0)

            # Separete the labels from the features and do one hot encoding for the neural network
            X_train, Y_train, _  = separete_data(train_data)

            if test:
                X_val, Y_val, y_test = separete_data(test_data)
            else:
                X_val, Y_val, y_test = separete_data(validation_data)

            # Initialize a new network per each new fold, all the networks are going to be the same
            NET_NAME = BASE_NAME + "_"+  str(i)
            net = Network(name = NET_NAME, base_name = BASE_NAME)
            net.init_self_net(HIDDEN_LAYERS, HIDDEN_NEURONS,drop = DROPOUT, beta1=BETA1, lr=LEARNING_RATE, activation = ACTIVATION, regularizer = REGULARIZER, regularization_penalty=REGULARIZATION_PENALTY)

            # Train the network and optimize it with using the validation set
            # Or train the final network and compute the actual test error
            net.train(X_train,Y_train,X_val,Y_val, epochs=EPOCHS, batch_size = N_BATCHES, save=save, file_path="models")
            y_pred = net.predict(X_val)
            # Plot confusion matrix
            plot_confusion_matrix(y_test, y_pred, BASE_NAME, normalize=True, fold=i)
            # and not normalized as well
            plot_confusion_matrix(y_test, y_pred, BASE_NAME, normalize=False, fold=i)
            calculate_scores(y_test, y_pred, BASE_NAME, fold=i)

            # Get the validation accracy for one fold
            accuracy = net.evaluate_model(X_val,Y_val)
            print("## Validation Accuracy:{} ##".format(accuracy))
            accuracies.append(accuracy)
        if test:
            break

    print("### Finished n-fold cross validation ###")
    final_accuracy = np.mean(np.array(accuracies))
    print("##### Average Accuracy:{} ######".format(final_accuracy))


if __name__ == "__main__":
    n_fold(test = TEST)
