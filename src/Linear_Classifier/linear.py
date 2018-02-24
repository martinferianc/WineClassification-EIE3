from sklearn import linear_model
import pickle
import os

import numpy as np

class LinearRegressionClassifier:
    """
    This is a wrapper for linear regression for classification
    """

    def __init__(self, name, base_name):
        self.base_name = base_name
        self.name = name
        self.clf = None

    # Trains the classifier
    def train(self,X_train,Y_train,loss="squared_loss",epochs=20,regularizer="l2",regularizer_penalty=0.0001, stop=0.001, save=False, file_path="models"):
        if X_train.size == 0 or Y_train.size == 0:
            raise EmptyDataError("No data to train on!")

        self.clf = linear_model.SGDClassifier(alpha=regularizer_penalty, average=False, class_weight=None, epsilon=0.1,
                                              learning_rate='optimal', loss=loss, max_iter=epochs,
                                              penalty=regularizer, tol=stop, verbose=5, warm_start=False)
        self.clf.fit(X_train,Y_train)
        if save:
            if not os.path.isdir(file_path+"/"+self.base_name+"/"+self.name):
                os.makedirs(file_path+"/"+self.base_name+"/"+self.name)
            self.save("{}/{}/{}/{}".format(file_path, self.base_name, self.name, self.name))

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
