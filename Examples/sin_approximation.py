import numpy as np
import random
import matplotlib.pyplot as plt


def gradient_descent(X,Y,lr, max_iter=1000, sc=0.001):
    W = np.ones(X.shape[1])
    i = 0
    previous_loss = 0
    current_loss = -1
    while i<max_iter and abs(previous_loss-current_loss)>0.001:
        h = np.dot(X,W)
        loss = h-Y
        current_loss = np.sum(loss**2)
        print("Iteration %d | Cost: %f" % (i, current_loss))
        gradient = (2/len(X))*np.dot(X.transpose(), loss)
        previous_loss = current_loss
        W-=lr*gradient
        h = np.dot(X,W)
        current_loss = np.sum((h - Y)**2)
        i+=1
    return W

def linear_regression(X,Y,reg=None):
    if reg is None:
        reg = 0
    W = np.matmul(np.linalg.inv(np.matmul(X.transpose(),X)+reg*np.identity(X.shape[1])),np.matmul(X.transpose(),Y))
    return W



def generate_data(N, power, var=0.1):
    X = np.zeros(shape=(N,power+1))
    Y = np.zeros(N)
    for i in range(N):
        x = random.uniform(0,1)
        y = np.sin(x*2*np.pi) + np.random.normal(0,var)
        X[i] = np.array([x**j for j in range(power+1)])
        Y[i]=y
    return X,Y
train = 100
X,Y = generate_data(1000, 120, 0.01)
#W = gradient_descent(X,Y,max_iter = 5000,lr=0.09,sc = 0.0001)
W = linear_regression(X[:train],Y[:train],0.001)
print(W)
x_test = [X[i] for i in range(len(X[train:]))]
y_test = []
for x in x_test:
    y_test.append(np.dot(W,x))

x_test = [x[1] for x in x_test]

plt.figure()
X = [x[1] for x in X]
plt.scatter(X,Y)

plt.scatter(x_test,y_test)
plt.show()
