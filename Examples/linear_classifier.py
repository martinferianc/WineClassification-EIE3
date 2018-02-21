import pandas as pd
import numpy as np
from random import randrange, randint
from sklearn import preprocessing
import matplotlib.pyplot as plt

#Replace address, destination and so on with numbers
def handle_non_numberical_data(data):
    columns = data.columns.values
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        if data[column].dtype != np.int64 and data[column].dtype!= np.float64:
            column_contents = data[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1
            data[column] = list(map(convert_to_int, data[column]))
    return data


#Preprocess the data and convert everything into numbers
def preprocess_data(input,validation):
    data = pd.read_csv(input)
    data.drop(['PassengerId'], 1 , inplace = True)
    data.convert_objects(convert_numeric=True)
    data.fillna(0, inplace=True)
    data = handle_non_numberical_data(data)
    X = np.array(data.drop(['Survived'],1)).astype(float)
    X = preprocessing.scale(X)
    Y = np.array(data['Survived'])
    Y[Y==0] = -1
    l = int(validation*len(X))
    test_X = X[:l]
    test_Y = Y[:l]
    train_X = X[l:]
    train_Y = Y[l:]
    return (train_X, train_Y, test_X, test_Y)

def generate_clustered_data(k=2,n=100,validation=0.1):
    X = []
    Y = []
    for i in range(k):
        for j in range(int(n/k)):
            x_1 = randrange(0,5) + 5*i
            x_2 = randrange(0,5) + 5*i
            X.append(np.array([x_1,x_2]))
            Y.append(-1 + i*2)
    X = np.array(X)
    Y = np.array(Y)
    l = int(validation*len(X))
    test_X = X[:l]
    test_Y = Y[:l]
    train_X = X[l:]
    train_Y = Y[l:]
    return (train_X, train_Y, test_X, test_Y)

class LinearClassifier:
    def __init__(self,threshold):
        self.W = None
        self.threshold = threshold
        self.iterations = 0

    def fit(self,X,Y,iterations):
        z = np.full((X.shape[0],1), self.threshold)
        X = np.append(X, z, axis=1)
        self.W = np.zeros(shape=X.shape[1])
        for i in range(iterations):
            error = False
            self.iterations+=1
            for index in range(len(X)):
                s = np.dot(self.W,X[index])
                y = np.sign(s)
                if y!=Y[index]:
                    error = True
                    self.W+=Y[index]*X[index]
            if error is False:
                break

    def test(self,X,Y):
        error = 0
        z = np.full((X.shape[0],1), self.threshold)
        X = np.append(X, z, axis=1)
        for index in range(X.shape[0]):
            s = np.dot(self.W,X[index])
            y = np.sign(s)
            if y!=Y[index]:
                error+=1
        print("Accuracy of this run: {}, Test set: {} examples, Needed iterations:{}".format(1-error/len(X),len(X),self.iterations))
        print("Weight vector: ")
        print(self.W)

    def get_weight(self):
        return self.W


#train_X, train_Y, test_X, test_Y = preprocess_data("../../data/titanic/train.csv",0.01)
train_X, train_Y, test_X, test_Y = generate_clustered_data(2,40,0.1)
c = LinearClassifier(1)
c.fit(train_X,train_Y, 100)
c.test(test_X, test_Y)
x = [i[0] for i in train_X]
y = [i[1] for i in train_X]
plt.scatter(x,y)
W = c.get_weight()
line_x = [randint(0,10) for x in range(10)]
line_y = [-(W[0]*line_x[i]+W[2])/W[1] for i in range(10)]
plt.scatter(line_x, line_y)


plt.show()
