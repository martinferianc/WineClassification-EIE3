import numpy as np
import random

# m denotes the number of examples here, not the number of features
def linear_regression(X,Y):
    W = np.matmul(np.linalg.inv(np.matmul(X.transpose(),X)),np.matmul(X.transpose(),Y))
    return W


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

# gen 100 points with a bias of 25 and 10 variance as a bit of noise
x, y = genData(100, 25, 10)
W = linear_regression(x,y)
print(W)
