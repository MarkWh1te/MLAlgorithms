import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

data = loadmat('ex3data1.mat')
print(data)

# load data
X = data['X']
y = data['y']
print(X.shape,y.shape)

# one-hot encoding
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(y)
print(y[0],y_onehot[0,:])

# activation function(sigmoid function)
def sigmoid(x):
    return 1/(1+np.exp(-x))

# feedforward
def feedforward(X,theta1,theta2):
    m,n = X.shape

    a1 = np.insert(X,0,values=np.ones(m),axis=1)
    z2 = a1*theta1.T
    a2 = np.insert(sigmoid(z2),0,values=np.ones(m),axis=1)
    z3 = a2*theta2.T
    a3 = sigmoid(z3)
    h = a3
    return a1,z2,a2,z3,h

# cost function
