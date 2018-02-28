from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from tflearn.optimizers import Adam
from tflearn import DNN
from tflearn import activations
import os

import numpy as np

class Network:
    """
    This is a wrapper class for the neural network structure
    """
    def __init__(self, name, base_name):
        self.base_name = base_name
        self.name = name
        self.model = None
        self.net = None

    #Initializes the network
    def init_self_net(self ,hidden_layers, hidden_neurons,drop=0.8,beta1=0.99,lr=0.001, activation="relu", n_features=11, n_classes=11, regularizer="", regularization_penalty=0.001):
        self.net = input_data(shape=[None, n_features], name='input')
        for i in range(hidden_layers):
            self.net = fully_connected(self.net, hidden_neurons, regularizer=regularizer, weight_decay=regularization_penalty)

            if activation == "elu":
                self.net = activations.elu(self.net)
            elif activation == "relu":
                self.net = activations.relu(self.net)
            elif activation == "tanh":
                self.net = activations.tanh(self.net)
            elif activation == "selu":
                self.net = activations.selu(self.net)
            self.net = dropout(self.net, drop)

        self.net = fully_connected(self.net, n_classes, activation='softmax')

        # Define the optimizer
        o = Adam(learning_rate=lr, beta1=beta1)
        self.net = regression(self.net, optimizer=o, loss='categorical_crossentropy', name='targets')

        return self.net
    # Trains the netowk and stores all the outputs and logs in the respective directories
    def train(self,X_train,Y_train,X_val,Y_val,epochs=5, batch_size = 64, save=False, file_path="models"):
        if self.net is None:
            raise EmptyNetError("No self.net to be trained!")
        self.model = DNN(self.net, tensorboard_verbose=3, tensorboard_dir="Neural_Network/logs/{}".format(self.base_name))
        self.model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=epochs, validation_set=({'input': X_val}, {'targets': Y_val}),
        snapshot_step=50, show_metric=True, batch_size=batch_size, run_id=self.name)
        if save:
            if not os.path.isdir("Neural_Network/"+file_path+"/"+self.base_name+"/"+self.name):
                os.makedirs("Neural_Network/"+file_path+"/"+self.base_name+"/"+self.name)
            self.save_net("Neural_Network/{}/{}/{}/{}".format(file_path, self.base_name, self.name, self.name))

    def evaluate_model(self, X,Y):
        if X.size == 0 or Y.size == 0:
            raise EmptyDataError("No data to train on!")
        if self.net is None or self.model is None:
            raise EmptyNetError("No self.net to be trained!")
        return self.model.evaluate(X,Y, batch_size=X.shape[0])

    def predict(self,X):
        if X.size == 0:
            raise EmptyDataError("No data to train on!")
        if self.net is None or self.model is None:
            raise EmptyNetError("No self.net to be trained!")
        labels = []
        for feature_set in X:
            labels.append(np.argmax(self.model.predict([feature_set])))
        return np.array(labels)

    def load_net(self, file_path):
        if self.net is None:
            raise EmptyNetError("No self.net to be loaded!")
        self.model = DNN(self.net)
        self.model.load(file_path)

    def save_net(self, file_path):
        if self.net is None or self.model is None:
            raise EmptyNetError("No self.net to be saved!")
        self.model.save(file_path)
