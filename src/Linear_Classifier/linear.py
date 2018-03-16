from sklearn import linear_model
import pickle
import os
import sys
import matplotlib.pyplot as plt
import io
import numpy as np
import copy
from scipy.signal import medfilt

class LinearRegressionClassifier:
    """
    This is a wrapper for linear regression for classification
    """

    def __init__(self, name, base_name):
        self.base_name = base_name
        self.name = name
        self.clf = None

    # Trains the classifier
    def train(self,X_train,Y_train,X_val, Y_val, loss="squared_loss",epochs=20,n_batches=10,regularizer="l2",regularizer_penalty=0.0001,learning_rate="optimal", stop=0.001, save=False, file_path="models"):
        if X_train.size == 0 or Y_train.size == 0:
            raise EmptyDataError("No data to train on!")

        self.accuracies_train = []
        self.accuracies_val = []

        # Separate the data into mini-batches
        X_train_batches = np.array_split(X_train, n_batches)
        Y_train_batches = np.array_split(Y_train, n_batches)


        old_stdout = sys.stdout
        sys.stdout = mystdout = io.StringIO()
        # Initialize the classifier
        self.clf = linear_model.SGDClassifier(alpha=regularizer_penalty, average=False, class_weight=None, epsilon=0.1,
                                              n_jobs = 1,learning_rate="optimal", eta0=learning_rate, loss=loss, max_iter=5, fit_intercept=False,
                                              penalty=regularizer, tol=stop, verbose=1, warm_start=True)

        # Train for the maximum number of epochs and iteration
        for i in range(epochs):
            for j in range(len(X_train_batches)):
                self.clf.partial_fit(X_train_batches[j],Y_train_batches[j], classes=np.unique(Y_train))
                accuracy_train = self.evaluate_model(X_train, Y_train)
                accuracy_val = self.evaluate_model(X_val, Y_val)
                self.accuracies_train.append(accuracy_train)
                self.accuracies_val.append(accuracy_val)

        # Extract the loss from the printfs
        sys.stdout = old_stdout
        loss_history = mystdout.getvalue()
        self.losses = []

        for line in loss_history.split('\n'):
            if(len(line.split("loss: ")) == 1):
                continue
            self.losses.append(float(line.split("loss: ")[-1]))
        # Losses for all 7 different classifiers are outputed therefore average the loss between all of them

        N = 7
        self.losses = np.convolve(self.losses, np.ones((N,))/N)[(N-1):]
        self.losses = self.losses[:-7]
        self.losses = self.losses[::N]

        if save:
            if not os.path.isdir("Linear_Classifier"+"/models/"+self.base_name+"/"+self.name):
                os.makedirs("Linear_Classifier"+"/models/"+self.base_name+"/"+self.name)
            self.save("Linear_Classifier/{}/{}/{}/{}".format(file_path, self.base_name, self.name, self.name))

    def evaluate_model(self, X,Y):
        if X.size == 0 or Y.size == 0:
            raise EmptyDataError("No data to train on!")
        if self.clf is None:
            raise EmptyClfError("No self.clf trained or loaded!")
        error = 0
        for i in range(X.shape[0]):
            label = self.clf.predict(X[i].reshape(1,-1))
            if label!=Y[i]:
                error+=1

        return (1-error/X.shape[0])

    def visualize(self, file_path=None):
        if not os.path.isdir("Linear_Classifier/{}/{}/{}".format("logs", self.base_name, self.name)):
            os.makedirs("Linear_Classifier/{}/{}/{}/".format("logs", self.base_name, self.name))

        # Visualize the loss function
        plt.figure()
        plt.plot(np.arange(len(self.losses)), self.losses, 'bo', np.arange(len(self.losses)), self.losses, 'k')
        plt.title("Plot of loss function for {}".format(self.name))
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.savefig("Linear_Classifier/{}/{}/{}/loss_{}.png".format("logs", self.base_name, self.name, self.name))
        plt.close()

        # Visualize the training accuracy
        plt.figure()
        plt.plot(np.arange(len(self.accuracies_train)), self.accuracies_train, 'bo', np.arange(len(self.accuracies_train)), self.accuracies_train, 'k')
        plt.title("Plot of training accuracy for {}".format(self.name))
        plt.xlabel("Iterations")
        plt.ylabel("Training accuracy")
        plt.grid(True)
        plt.savefig("Linear_Classifier/{}/{}/{}/t_error_{}.png".format("logs", self.base_name, self.name, self.name))
        plt.close()

        # Visualize the validation accuracy
        plt.figure()
        plt.plot(np.arange(len(self.accuracies_val)), self.accuracies_val, 'bo', np.arange(len(self.accuracies_val)), self.accuracies_val, 'k')
        plt.title("Plot of validation accuracy for {}".format(self.name))
        plt.xlabel("Iterations")
        plt.ylabel("Validation accuracy")
        plt.grid(True)
        plt.savefig("Linear_Classifier/{}/{}/{}/v_error_{}.png".format("logs", self.base_name, self.name, self.name))
        plt.close()

        # Visualize the validation and training accuracy together
        plt.figure()
        plt.plot(np.arange(len(self.accuracies_train)), self.accuracies_train, label="Training Accuracy")
        plt.plot(np.arange(len(self.accuracies_val)), self.accuracies_val, label = "Validation Accuracy")
        plt.title("Plot of T. accuracy and V. accuracy for {}".format(self.name))
        plt.xlabel("Iterations")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.grid(True)
        plt.savefig("Linear_Classifier/{}/{}/{}/t_v_error_{}.png".format("logs", self.base_name, self.name, self.name))
        plt.close()


    def predict(self,X):
        if X.size == 0:
            raise EmptyDataError("No data to train on!")
        if self.clf is None:
            raise EmptyClfError("No self.clf trained or loaded!")
        labels = []
        for feature_set in X:
            labels.append(self.clf.predict(feature_set.reshape(1,-1)))
        return np.array(labels)

    def save(self, file_path):
        with open(file_path, 'wb') as handle:
            pickle.dump(self.clf, handle, protocol=pickle.HIGHEST_PROTOCOL)
            return True

    def load(self, file_path):
        with open(file_path, 'rb') as handle:
            self.clf = pickle.load(handle)
        return True
