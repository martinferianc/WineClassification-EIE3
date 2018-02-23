import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

class BinaryLinearRegression():
    """
    A linear regression class for a binar linear classifier
    """

    def __init__(self, name):
        self.W = None
        self.name = name

    # This function adds a bias of ones to the data
    def __preprocess_data(self, data):
        features = [x[:-1] for x in data]
        labels = [x[-1] for x in data]
        n_training_samples = features.shape[0]
        n_dim = features.shape[1]
        f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
        l = np.reshape(labels,[n_training_samples,1])
        return f, l

    # Performs the main training
    def train(self, data, lr, epochs,regularizer=None, penalty=None, save=True):
        # Add a bias term to the data
        f, l = self.__preprocess_data(data)

        # Initialize an empty cost history
        loss_history = np.empty(shape=[1],dtype=float)
        n_dim = f.shape[1]
        X = tf.placeholder(tf.float32,[None,n_dim])
        Y = tf.placeholder(tf.float32,[None,1])
        self.W = tf.Variable(tf.ones([n_dim,1]))

        # How is the label calculated
        y_ = tf.sign(tf.matmul(X, self.W))

        # Define loss
        loss = tf.reduce_mean(tf.square(y_ - Y))

        # Decide it to save the weights or not
        saver = None
        if save:
            saver = tf.train.Saver({"W": self.W})

        #Define regularizer
        regularization_penalty = 0
        if regularizer == "L1":
            regularizer = tf.contrib.layers.l1_regularizer(scale=penalty, scope=None)
            regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, self.W)

        elif regularizer = "L2":
            regularizer = tf.contrib.layers.l2_regularizer(scale=penalty, scope=None)
            regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer,self.W)

        # Define the total loss andregularizer if any
        regularized_loss = total_loss + regularization_penalty

        # Definie the training step and the optimizer
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(regularized_loss)

        # Begin training
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for e in range(epochs):
                sess.run(training_step,feed_dict={X:f,Y:l})
                loss_history = np.append(loss_history,sess.run(regularized_loss,feed_dict={X: f,Y: l}))

            if save:
                save_path = saver.save(sess, "/models/{}.ckpt".format(self.name))
        return loss_history


    def test(data, file_path="/models/{}.ckpt".format(self.name)):
        # Add a bias term to the data
        f, l = self.__preprocess_data(data)

        tf.reset_default_graph()
        self.W = tf.Variable(tf.ones([n_dim,1]))

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver({"W": self.W})

        with tf.Session() as sess:
          # Restore variables from disk.
          saver.restore(sess, file_path)
          #
          sess.run(training_step,feed_dict={X:f,Y:l})
