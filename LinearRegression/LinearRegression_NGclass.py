import numpy as np
import pandas as pd


data = pd.read_csv("ex1data2.txt",header=None,names=["Size", "Bedrooms", "Price"])
print(data.head())
print(data.describe())


# feature normalize
def normalize(df):
    return (df-df.mean())/df.std()

data = normalize(data)

# add constant column to x(to accommodate the θ0 intercept term)
data.insert(0, 'Ones', 1)
data.head()

# variable initialization
data_matrix = np.matrix(data.values)

X = data_matrix[:,0:data_matrix.shape[1]-1]
y = data_matrix[:,data_matrix.shape[1]-1:data_matrix.shape[1]]
theta = np.matrix(np.array([0]*X.shape[1]))

# def normalize(matrix):
    # return (matrix-np.mean(matrix,axis=0))/np.std(matrix,axis=0)

# X = normalize(X)

print(X.shape,y.shape,theta.shape)

def computeCost(X,y,theta):
    hypotheses = X * theta.T
    inner_product = np.power(hypotheses-y,2)
    return np.sum(inner_product)/(2*X.shape[0])

# setting learning rate and iteration times
alpha = 0.01
iters = 1000

def gradientDescent(X,y,theta,alpha,iters):
    print(theta.shape)
    m,n = X.shape
    tmp_theta = np.matrix(np.zeros(theta.shape))
    for i in range(iters):
        print(i)
        error = (X* theta.T) - y
        for j in range(theta.shape[1]):
            gradient = np.multiply(error,X[:,j])/X.shape[0]
            tmp_theta[0,j] = theta[0,j] - alpha * sum( gradient )
        theta = tmp_theta
        print(computeCost(X,y,theta))
    return theta

theta = gradientDescent(X, y, theta, alpha, iters)
print(computeCost(X,y,theta))

