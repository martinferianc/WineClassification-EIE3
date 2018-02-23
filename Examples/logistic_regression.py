import numpy as np
import scipy.io
import random
import math
import matplotlib.pyplot as plt
import copy

class LogisticRegression():

    def __init__(self, lr, max_iterations = 1000, sc = 0.001):
        self.lr = lr
        self.max_iterations = max_iterations
        self.sc = sc
        self.W = None

    def compute_loss(self,X,Y):
        if self.W is None:
            self.W = np.full(X.shape[1], 1//len(X),dtype=np.float64)
        R = 0
        for i in range(len(X)):
            R+=(Y[i]*X[i])/(1+np.exp(Y[i]*np.matmul(self.W.transpose(),X[i])))
        R*=(-1/len(X))
        return R

    def train(self,X,Y, mode="sgd",n=10,epochs = None):
        self.W = np.full(X.shape[1], float(1)/float(len(X)),dtype=np.float64)
        previous_loss = -1
        current_loss = 1
        i = 0
        loss = []
        x = X
        y = Y

        if mode == "sgd":
            x = np.split(x,n)
            y = np.split(y,n)

        # Stochastic gradient method
        if epochs is not None and n is not None and mode == "sgd":
            for i in range(epochs):
                print("Epochs: {}".format(i))
                for j in range(n):
                    current_loss = self.compute_loss(x[j],y[j])
                    print("Iteration: {}".format(j))
                    print("Previous Loss: {}".format(np.sum(current_loss)))
                    #print("Previous weights: {}".format(self.W))
                    self.W -= self.lr*current_loss
                    #print("New weights: {}".format(self.W))
                    previous_loss = current_loss
                    current_loss = self.compute_loss(x[j],y[j])
                    print("New Loss: {}".format(np.sum(current_loss)))
                loss.append((i,np.sum(current_loss)))
            return loss



        while abs(np.sum(previous_loss-current_loss))>self.sc and i<self.max_iterations:
            current_loss = self.compute_loss(X,Y)
            loss.append((i,np.sum(current_loss)))
            print("Iteration: {}".format(i))
            print("Previous Loss: {}".format(np.sum(current_loss)))
            #print("Previous weights: {}".format(self.W))
            self.W -= self.lr*current_loss
            #print("New weights: {}".format(self.W))
            previous_loss = current_loss
            current_loss = self.compute_loss(X,Y)
            print("New Loss: {}".format(np.sum(current_loss)))
            i+=1
        return loss


    def test(self,X,Y):
        error = 0
        for i in range(len(X)):
            l = np.sign(np.sum(X[i] * self.W))
            if l!=Y[i][0]:
                error+=1
        print("Accuracy: {}".format(float(1)-float(error/len(X))))


def genData(numPoints, bias, variance):
    x = np.zeros(shape=(numPoints, 2))
    y = np.zeros(shape=numPoints)
    # basically a straight line
    for i in range(0, numPoints):
        # bias feature
        x[i][0] = 1
        x[i][1] = i
        # our target variable
        y[i] = (i + bias) + random.uniform(0, 1) * variance
    return x, y



def load_data(file_path):
    data = scipy.io.loadmat(file_path)
    X = data["x"][:100]
    Y = np.array(data["y"][:100],dtype=np.int32)
    a = np.ones((len(X),1))
    X = np.hstack((X,a))
    for y in Y:
        if y[0] != 1:
            y[0] = -1
    return X,Y
if __name__ == '__main__':
        X,Y = load_data("../data/data_doc.mat")
        c = LinearRegression(0.05, 4000, 0.0001)
        loss = c.train(X,Y)
        l_x = [x[0] for x in loss]
        l_y = [y[1] for y in loss]
        c.test(X,Y)
        plt.figure()
        plt.scatter(l_x,l_y)
        plt.show()
        #print(Y)
