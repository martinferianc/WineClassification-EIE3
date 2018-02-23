import numpy as np
import copy
import os
from tqdm import tqdm

BASE_DIR = "data/"

# Load the data and convert it into numpy matrices

# Load the dataset in csv format and set the delimiter
def load_data(file_path):
    data = np.genfromtxt(file_path, delimiter=";")
    return data


def load_data_set(data_quality, fold=0, d_set="training", base_dir=BASE_DIR + "processed"):
    file_path = base_dir + "/{0}/{1}{2}.npy".format(data_quality, fold, d_set)
    print("Loading file: {0}".format(file_path))
    return np.load(file_path)

def normalize(data):
    # Transpose the data so that we can operate on columns as rows
    data = np.transpose(data)

    # For each row normalize it so that mean is zero and has a unit std
    # Do not normalize the labels!
    for i in tqdm(range(len(data)-1)):
        data[i] = (data[i] - np.mean(data[i]))/np.std(data[i])

    # Transpose the data back
    data = np.transpose(data)

    return data


# Split the data into n folds of training, test and validation data
def n_fold(data, target_dir, n=10, validation=0.1, test=0.1):
    # Load the target data into two vars

    # Reshuffle the data to avoid any clusters
    data = np.take(data,np.random.permutation(data.shape[0]),axis=0,out=data)

    #Normalize all the data
    data = normalize(data)
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

def build_data_sets(types = ["red", "white"]):
    print("### Beginning preprocessing ###")

    for t in types:
        print("## Beginning preproecessing {} data ##".format(t))
        if not os.path.exists(os.path.join(BASE_DIR, "processed", t)):
            os.makedirs(os.path.join(BASE_DIR, "processed", t))
        file_path = (BASE_DIR + "raw/winequality-{}.csv").format(t)

        print("Loading from file_path {}".format(file_path))
        data = load_data(file_path)
        n_fold(copy.deepcopy(data), (BASE_DIR + "processed/{0}/").format(t))
        print("## Finished preproecessing {} data ##".format(t))
    print("### Finished preprocessing ###")

if __name__ == "__main__":
    build_data_sets()
