import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import test as tst
from exceptions import RateCalcError, OutofBoundsError

class Postprocessing:

    def __init__(self, N):
        self.confusion_matrix = np.zeros((N, N))  # Assumes no data is skipped in x_actu
        self.mat_size = N
        self.actual = []

    ########## Cross validation: Total error estimate = arithmetic mean of error(D) ###########
    ########## Generates a confusion matrix ###########
    # Input: List of actual and predicted values
    # Output: Confusion matrix

    # Without the library
    def confusion_gen(self, x_actu, x_pred):
        N = len(set(x_actu))
        self.actual.extend(x_actu)

        if N > self.mat_size:
            raise OutofBoundsError("Error in generating confusion matrix: more labels than expected")

        for a, p in zip(x_actu, x_pred):
            self.confusion_matrix[a-1][p-1] += 1

        return self.confusion_matrix


    ########## Generate a confusion matrix plot #########
    # Input: Confusion Matrix
    # Output: Plots the confusion matrix
    def plot_confusion_matrix(self, title='Confusion matrix', cmap=plt.cm.Blues):

        norm_conf = []
        for i in self.confusion_matrix:
            a = 0
            tmp_arr = []
            a = sum(i, 0)
            for j in i:
                tmp_arr.append(float(j)/float(a))
            norm_conf.append(tmp_arr)

        row_sums = self.confusion_matrix.sum(axis=1)
        norm_matrix = self.confusion_matrix / row_sums[:, np.newaxis]
        norm_matrix = np.around(norm_matrix,decimals = 2)

        fig = plt.figure()
        plt.clf()
        ax = fig.add_subplot(111)
        ax.set_aspect(1)
        res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                        interpolation='nearest')

        width, height = norm_matrix.shape

        for x in range(width):
            for y in range(height):
                ax.annotate(str(norm_matrix[x][y]), xy=(y, x),
                            horizontalalignment='center',
                            verticalalignment='center')

        cb = fig.colorbar(res)
        alphabet = '012345678910'
        plt.xticks(range(width), alphabet[:width])
        plt.xlabel("Actual Classes")
        plt.yticks(range(height), alphabet[:height])
        plt.ylabel("Predicted Classes")
        plt.savefig('{}.png'.format(title), format='png')


    ######## Get the actual labels in a similar order #########
    # Input: array of x, y (nx45, nx1)
    # Output: Single list of all labels
    def read_true_vals(self, original_list):
        labels_list = []

        for i in range(len(original_list)):
            labels_list.append(original_list[i][1])

        return labels_list


    ########## Returns calculated rates ###########
    # Input: The confusion matrix for calculations
    # Output: A dictionary of precision, recall and error
    def r_calc(self,class_no):
        if class_no >= len(self.confusion_matrix):
            raise OutofBoundsError("Error in generating confusion matrix size")

        fp = fn = tp = tn = 0
        tp = self.confusion_matrix[class_no][class_no]

        for i in range(len(self.confusion_matrix)):
            if i != class_no:
                fp += self.confusion_matrix[i][class_no]
                fn += self.confusion_matrix[class_no][i]
                for j in range(len(self.confusion_matrix)):
                    if j != class_no:
                        tn += self.confusion_matrix[i][j]

        if (tp + fn == 0 or tp + fp == 0):
            raise RateCalcError("Division by zero error, both tp and fp are zero")

        recall_r = (tp / (tp + fn)) * 100
        precision_r = (tp / (tp + fp)) * 100
        classif_rate = (tp + tn) / (tp + fp + fn + tn) * 100
        return {'recall_rate': recall_r, 'precision_rate': precision_r, 'classif_rate': classif_rate}


    ########## F(a) measures ###########
    # Input: All rates
    # Output: A dictionary of f1 for each class
    def f_calc(self, comb_rates, a):
        precision = comb_rates['precision_rate']
        recall = comb_rates['recall_rate']

        if precision + recall == 0:
            raise RateCalcError("Division by zero error, both precision and recall rates are zero")

        f_a = (1 + a**2) * ((precision * recall) / (a**2 * precision + recall))
        comb_rates['F_a'] = f_a
        return comb_rates


    def print_stats(self):
        print("Confusion Matrix:", self.confusion_matrix, sep="\n")
        for i in range(len(self.confusion_matrix)):
            td = self.r_calc(i)
            td = self.f_calc(td, 1)
            print("\nEmotion: ", i+1, "\nRecall Rate:", td['recall_rate'], "\nPrecision Rate:", td['precision_rate'],
              "\nClassification Rate:", td['classif_rate'], "\nF_a:", td['F_a'], sep=" ")
