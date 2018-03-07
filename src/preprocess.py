import numpy as np
import copy
import os
from tqdm import tqdm
import pandas
from string import ascii_letters
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = "data/"


# Load the dataset in csv format and set the delimiter
def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=";")
    return data

# Analyzes the data and generates the covariance matrix
def analyze_data(data):
    data = copy.deepcopy(data)
    features = np.array(copy.deepcopy([x[:-1] for x in data]), dtype=np.float64)

    d = pd.DataFrame(data=features)
    metrics = d.describe().loc[['mean','std','min','max']]
    metrics = metrics.transpose()
    print(metrics.to_latex())

    data = normalize(data)
    features = np.array(copy.deepcopy([x[:-1] for x in data]), dtype=np.float64)
    labels = np.array(copy.deepcopy([x[-1] for x in data]), dtype=np.int32)
    print("Number of elements per class:")
    l, c = np.unique(labels, return_counts = True)
    print("{}, {}".format(l,c))


    plt.hist(labels, bins = l)
    plt.title("Class histogram")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig("../Figures/{}".format("Class_Histogram.png"))
    plt.close()

    sns.set(style="white")

    d = pd.DataFrame(data=features)

    # Compute the correlation matrix
    corr = d.corr()

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, fontsize=13);
    plt.yticks(range(len(corr.columns)), corr.columns, fontsize=13);

    plt.title("Features",fontsize=20)
    plt.xlabel("Correlation matrix", fontsize=30)
    plt.ylabel("Features",fontsize=20)
    plt.savefig("../Figures/{}".format("Correlation_matrix.png"))

    plt.close()


# Loads a particular dataset which is needed at a time
def load_data_set(data_quality, fold=0, d_set="training", base_dir=BASE_DIR + "processed"):
    file_path = base_dir + "/{0}/{1}{2}.npy".format(data_quality, fold, d_set)
    print("Loading file: {0}".format(file_path))
    return np.load(file_path)

def normalize(data, zero_to_one=False):
    # Transpose the data so that we can operate on columns as rows
    data = np.transpose(data)

    # For each row normalize it so that mean is zero and has a unit std
    # Do not normalize the labels!
    for i in tqdm(range(len(data)-1)):
        data[i] = (data[i] - np.mean(data[i]))/np.std(data[i])
    if zero_to_one:
        for i in tqdm(range(len(data)-1)):
            mi = np.amin(data[i])
            data[i] -= mi
        for i in tqdm(range(len(data)-1)):
            mx = np.amax(data[i])
            data[i] /= mx

    # Transpose the data back
    data = np.transpose(data)

    return data


# Split the data into n folds of training, test and validation data
def n_fold(data, target_dir, n=10, validation=0.1, test=0.1):
    # Reshuffle the data to avoid any clusters
    np.random.shuffle(data)

    tseg = int((len(data)-1) * test)
    vseg = int((len(data)-1 - tseg) * validation)
    for i in range(0, n):

        # Split the data into test and non-test data
        tsplit_start = (i * tseg) % len(data)
        tsplit_end = (i * tseg + tseg) % len(data)
        test_data = data[tsplit_start:tsplit_end, :]

        # Split remaining data into training and validation
        r_data = np.vstack((data[0:tsplit_start, :], data[tsplit_end:, :]))
        vsplit_start = (i * vseg) % len(r_data)
        vsplit_end = (i*vseg + vseg)  % len(r_data)
        validation_data = r_data[vsplit_start:vsplit_end, :]
        training_data = np.vstack((r_data[0:vsplit_start, :], r_data[vsplit_end:, :]))

        # Print the shape of the data to check if split was done correctly
        print("Test data is {0}.".format(test_data.shape))
        print("Validation data is {0}.".format(validation_data.shape))
        print("Training data is {0}.".format(training_data.shape))
        np.save(target_dir + "{0}_test.npy".format(i), test_data)
        np.save(target_dir + "{0}_training.npy".format(i), training_data)
        np.save(target_dir + "{0}_validation.npy".format(i), validation_data)

# Builds the datasets and normalizes them
def build_data_sets(types = ["red", "white"]):
    print("### Beginning preprocessing ###")
    file_path_red = (BASE_DIR + "raw/winequality-red.csv")
    file_path_white = (BASE_DIR + "raw/winequality-white.csv")

    print("Loading files")
    data_red = load_data(file_path_red)
    #Denote red wine
    data_red = np.insert(data_red, 0, values=0, axis=1)

    data_white = load_data(file_path_white)
    #Denote white wine
    data_white = np.insert(data_white, 0, values=1, axis=1)

    # Join the red and white wines togeter
    data = np.concatenate((data_red,data_white), axis = 0)

    analyze_data(data)

    #Normalize all the data
    data = normalize(data)

    n_fold(copy.deepcopy(data), BASE_DIR + "processed/")
    print("### Finished preprocessing ###")

if __name__ == "__main__":
    build_data_sets()
