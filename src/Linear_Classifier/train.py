import os
import sys
import copy
import numpy as np

from visualize import visualize
from linear import LinearRegressionClassifier
from postprocess import plot_confusion_matrix, calculate_scores

# Initialize the basic parameters of the network
LEARNING_RATE = 0.01
LOSS = "squared_loss"
STOP = 0.0000001
REGULARIZER = "L2"
REGULARIZATION_PENALTY = 0.01
EPOCHS = 1
N_BATCHES = 5
TEST = False
BASE_NAME = str(LEARNING_RATE) + "_" + str(LOSS) + "_" + str(STOP) + "_" + str(REGULARIZER) + "_" + str(REGULARIZATION_PENALTY) + "_" + str(EPOCHS) + "_" + str(N_BATCHES)

# Separete the labels from the features
def separete_data(data):
    features = np.array(copy.deepcopy([x[:-1] for x in data]), dtype=np.float64)
    labels = np.array(copy.deepcopy([x[-1] for x in data]), dtype=np.int32)
    return (features, labels)

# This function performs training for the specific network and performs n_fold cross validation
def n_fold(n_folds = 10, save = True, test = False):
    accuracies_val = []
    accuracies_train = []
    accuracies = []
    losses = []

    print("### Beginning n-fold cross validation with parameters: ###")
    print("## Classifier name: ##".format(BASE_NAME))
    print("## Learning rate:{} ##".format(LEARNING_RATE))
    print("## Epochs:{} ##".format(EPOCHS))
    print("## Regularizer:{} ##".format(REGULARIZER))
    print("## Regularizer penalty:{} ##".format(REGULARIZATION_PENALTY))

    if not os.path.isdir("models/"+BASE_NAME):
        os.makedirs("models/"+BASE_NAME)

    for i in range(0,n_folds):
        print("## Fold:{} ##".format(i))
        # Begin each training completely separetely

        # Load all the data into the work place
        train_data = np.load(os.path.join("..", "data", "processed", "{}_training.npy".format(i)))
        validation_data = np.load(os.path.join("..", "data", "processed", "{}_validation.npy".format(i)))
        test_data = np.load(os.path.join("..", "data", "processed",  "{}_test.npy".format(i)))
        if test:
            # Join training and validation data for the final test
            train_data = np.concatenate((train_data, validation_data), axis=0)

        # Separete the labels from the features and do one hot encoding for the neural network
        X_train, Y_train = separete_data(train_data)

        if test:
            X_val, Y_val = separete_data(test_data)
        else:
            X_val, Y_val = separete_data(validation_data)

        # Initialize a new classifier per each new fold, all the classifiers are going to have the same parameters
        CLF_NAME = BASE_NAME + "_"+  str(i)
        clf = LinearRegressionClassifier(name = CLF_NAME, base_name = BASE_NAME)

        # Train the classifier
        clf.train(X_train,Y_train,X_val,Y_val,n_batches = N_BATCHES, epochs=EPOCHS, loss = LOSS, regularizer = REGULARIZER, regularizer_penalty = REGULARIZATION_PENALTY, stop = STOP, save=save, file_path="models")
        clf.visualize()
        # Copy the training accuracy, loss and validation accuracy per fold
        accuracies_val.append(copy.deepcopy(clf.accuracies_val))
        accuracies_train.append(copy.deepcopy(clf.accuracies_train))
        losses.append(copy.deepcopy(clf.losses))

        # Get the validation accracy for one fold
        accuracy = clf.evaluate_model(X_val,Y_val)
        y_pred = clf.predict(X_val)

        # Plot confusion matrix
        plot_confusion_matrix(Y_val, y_pred, BASE_NAME, normalize=True, fold=i)
        # and not normalized as well
        plot_confusion_matrix(Y_val, y_pred, BASE_NAME, normalize=False, fold=i)
        calculate_scores(Y_val, y_pred, BASE_NAME, fold=i)
        print("## Validation Accuracy:{} ##".format(accuracy))
        accuracies.append(accuracy)
        # Delete the previous classifier to avoid retraining
        del clf
        if test:
            break

    visualize(accuracies_train, accuracies_val, losses, BASE_NAME)
    print("### Finished n-fold cross validation ###")
    final_accuracy = np.mean(np.array(accuracies))
    print("##### Average Accuracy:{} ######".format(final_accuracy))


if __name__ == "__main__":
    n_fold(test = TEST)
