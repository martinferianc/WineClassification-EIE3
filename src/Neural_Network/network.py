from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.optimizers import Adam
from tflearn.layers.estimator import regression
from tflearn import DNN
import os

import numpy as np

class Network:

    def __init__(self, name):
        self.name = name

    #Initializes the self.net
    def init_self_net(self ,hidden_layers, hidden_neurons, activation="relu", n_features=12, n_classes=11, regularizer=""):
        self.net = input_data(shape=[None, n_features], name='input')
        for i in range(hidden_layers):
            self.net = fully_connected(self.net, hidden_neurons, activation=activation, regularizer=regularizer)
            self.net = dropout(self.net, 0.8)
        self.net = fully_connected(self.net, n_classes, activation='softmax')

        # Define the optimizer
        o = Adam(learning_rate=lr, beta1=0.99)
        self.net = regression(self.net, optimizer=o, loss='categorical_crossentropy', name='targets')
        return self.net

    def train(self,X_train,Y_train,X_val,Y_val, batch_size = 64, save=False, file_path="models"):
        if self.net is None:
            raise EmptyNetError("No self.net to be trained!")
        self.model = DNN(self.net, tensorboard_verbose=3, tensorboard_dir="logs/{}".format(self.name))
        self.model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=self.epochs, validation_set=({'input': X_val}, {'targets': Y_val}),
        snapshot_step=50, show_metric=True, batch_size=batch_size, run_id=self.name)
        if save:
            if not os.path.isdir(file_path):
                os.makedirs(file_path)
            self.save_net("{}/{}/{}".format(file_path, self.name, self.name))

    def test(self, feature_vector):
        if self.model is None:
            raise EmptyNetError("No moddel to be tested!")
        labels = self.model.predict([feature_vector])
        return (np.argmax(labels[0])+1)


    def load_net(self, file_path):
        if self.net is None:
            raise EmptyNetError("No self.net to be loaded!")
        self.model = DNN(self.net)
        self.model.load(file_path)

    def save_net(self, file_path):
        if self.net is None:
            raise EmptyNetError("No self.net to be saved!")
        self.model.save(file_path)
