# splitting data
from time import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

before = time()
data = loadmat('ex4data1')

X = data['X']
X = np.insert(X, 0, 1, 1)
hidden_units = 30
iters = 1500
lamda = 0
alpha = 3

def changed(yy) :
    m = yy.shape[0]
    newy = np.zeros((m,10))
    for i in range(m) :
        newy[i][yy[i][0]] = 1
    return newy

y10 = changed(data['y'] - 1)

class NeuralNetwork() :
    def __init__(self, X, h, y, alpha, lamda) :
        np.random.seed(2)
        
        self.alpha = alpha
        self.lamda = lamda
        self.m = X.shape[0]
        self.f = X.shape[1]
        self.c = y.shape[1]
        self.h = h
        
        self.input = X
        self.y = y
        
        # epsilon = sqrt(6)/sqrt(number of units in L layer + number of units in L+1 layer)
        self.weights1 = np.random.uniform(-(np.sqrt(6)/np.sqrt(self.f + self.h)) , np.sqrt(6)/np.sqrt(self.f + self.h), (self.f, self.h))
        self.weights2 = np.random.uniform(-(np.sqrt(6)/np.sqrt(self.h + self.c)) , np.sqrt(6)/np.sqrt(self.h + self.c), (self.h + 1, self.c))
        
    def sigmoid(self, z) :
        return 1/(1 + np.exp(-z))

    def sigmoid_der(self, a) :
        return a * (1 - a)
    
    def foreward(self) :
        self.a2 = np.hstack((np.ones((self.m, 1)), self.sigmoid(self.input @ self.weights1)))
        
        self.a3 = self.sigmoid(self.a2 @ self.weights2)
        
    def cost(self) :
        
        first = np.multiply(self.y, np.log(self.a3))
        second = np.multiply(1 - self.y, np.log(1 - self.a3))
        reg = (self.lamda/(2*self.m)) * (np.sum(np.power(self.weights1[1:],2)) + np.sum(np.power(self.weights2[1:],2)))
        
        self.J = np.sum(first + second)/-self.m + reg

    def backprop(self) :
        delta2 = np.zeros((self.h + 1, self.c))
        delta1 = np.zeros((self.f, self.h))
        
        small_delta3 = self.a3 - self.y
        
        mult = small_delta3 @ self.weights2.T
        small_delta2 = np.multiply(mult, self.sigmoid_der(self.a2))
        

        delta2 += self.a2.T @ small_delta3
        delta1 += self.input.T @ small_delta2[:,1:]
        
        
        D2 = delta2/float(self.m)
        D2[1:] += self.lamda/self.m * self.weights2[1:]
        D1 = delta1/float(self.m)
        D1[1:] += self.lamda/self.m * self.weights1[1:]
        
        
        self.weights2 -= self.alpha*D2
        self.weights1 -= self.alpha*D1
        
    def predict(self, X) :
        pred1 = self.sigmoid(X @ self.weights1)
        pred1 = np.insert(pred1, 0, 1, 1)
        pred2 = self.sigmoid(pred1 @ self.weights2)
        pred2 = np.where(pred2 == np.max(pred2, 1)[:, np.newaxis], 1, 0)
        return pred2
    

nn = NeuralNetwork(X, hidden_units, y10, alpha, lamda)

cost_list = []
for i in range(iters) :
    nn.foreward()
    nn.cost()
    cost_list.append(nn.J)
    nn.backprop()

plt.plot(cost_list)
print('Cost Before Training :', cost_list[0])
print('Cost After Training :', cost_list[-1])

prediction = nn.predict(X)
print('Accuracy :', (np.sum((nn.y == prediction).all(axis = 1))/nn.m)*100)

after = time()
print('Time to Process :', after - before)

