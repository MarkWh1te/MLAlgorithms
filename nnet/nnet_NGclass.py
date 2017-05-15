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

# Sigmoid gradient
def sigmoidGradient(x):
    return (sigmoid(x)*(1-sigmoid))

# feedforward
def forward_propagate(X,theta1,theta2):
    m,n = X.shape

    a1 = np.insert(X,0,values=np.ones(m),axis=1)
    z2 = a1*theta1.T
    a2 = np.insert(sigmoid(z2),0,values=np.ones(m),axis=1)
    z3 = a2*theta2.T
    a3 = sigmoid(z3)
    h = a3
    return a1,z2,a2,z3,h

# cost function
def cost(params,input_size,hidden_size,num_labels,X,y,learning_rate):
    m = X.shape[0]
    x = np.matrix(X)
    y = np.matrix(y)

    # unroll params
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
   

    # run feed forward pass
    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)


    # compute cost with without for loop
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)

    J = J/m

    J += (float(learning_rate) / (2 * m)) * (np.sum(np.power(theta1[:,1:], 2)) + np.sum(np.power(theta2[:,1:], 2)))
    return J

input_size = 400
hidden_size = 25
num_labels = 10
learning_rate = 1

params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.25


m = X.shape[0]
X = np.matrix(X)
y = np.matrix(y)



theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))

a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)

# def backprop()





















