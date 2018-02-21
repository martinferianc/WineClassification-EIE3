from __future__ import print_function

import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import scipy.io

def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    X_train =  np.array(data["x"][:900])
    Y_train = np.array(data["y"][:900],dtype=np.int64)
    X_test =  np.array(data["x"][900:])
    Y_test = np.array(data["y"][900:],dtype=np.int64)

    return X_train, Y_train, X_test, Y_test

# Preprocess data
X_train, Y_train, X_test, Y_test = load_data("../data/data_doc.mat")

Y_pre = np.zeros(shape=(900,6))
for i in range(Y_train.shape[0]):
    Y_pre[i][Y_train[i]-1] = 1

Y_val = np.zeros(shape=(Y_test.shape[0],6))
for i in range(Y_test.shape[0]):
    Y_val[i][Y_test[i]-1] = 1



net = input_data(shape=[None, 45], name='input')
net = fully_connected(net, 100, activation='relu')
net = dropout(net, 0.9)
net = fully_connected(net, 80, activation='relu')
net = dropout(net, 0.8)
net = fully_connected(net, 60, activation='relu')
net = dropout(net, 0.7)
net = fully_connected(net, 40, activation='relu')
net = dropout(net, 0.6)
net = fully_connected(net, 6, activation='softmax')
momentum = tflearn.optimizers.Momentum(learning_rate=0.01, lr_decay=0.96, decay_step=300)
net = regression(net, optimizer=momentum, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(net)
model.fit({'input': X_train}, {'targets': Y_pre}, n_epoch=10, validation_set=({'input': X_test}, {'targets': Y_val}),
    snapshot_step=50, show_metric=True, run_id='potato')

print(model.predict([X_test[0]]))
print(np.argmax(model.predict([X_test[0]])[0]))
print(Y_val[0])
