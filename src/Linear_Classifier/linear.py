from binary import BinaryLinearRegression
from exceptions import NoModeError
import pickle
import os
import numpy as np
import copy
import random


class LinearRegression:
    """
    A wrapper class to store the linear regressors
    """

    def __init__(self):
        self.regressors = []
        pass

    # Get scores from all the emotion trees
    def __test(self, feature_set, n_classes = 11):
        test_results = []
        for i in range(len(self.trees)):
            # Appends the label of the regressor
            test_results.append((self.regressors[i].test(feature_set), i + 1))

        test_results_ones = []
        # Filters out -1 labels
        for result in test_results:
            # Label
            if result[0] == 1:
                test_results_ones.append(result)

        # If no regressor could classify the data pick a label at random
        if len(test_results_ones) == 0:
            return np.random.randint(0, n_classes)

        # If 2 or more regressos classified the data, pick one at random
        elif len(test_results_ones)>2:
            choice = random.choice(test_results_ones)
            # Return the label
            return choice[1]
        else:
            return test_results_ones[0][1]

    # Tests all the data in bulk
    def test(self, data):
        labels = []
        for row in data:
            labels.append(self.__test(row))
        return np.array(labels)

    # Save the classifier
    def save(self, file_path):
        with open(file_path, 'wb') as handle:
            pickle.dump(self.regressors, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return True

    # Loads the classifier
    def load(self, file_path):
        with open(file_path, 'rb') as handle:
            self.regressors = pickle.load(handle)
        return True
