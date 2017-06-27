import numpy as np
import pandas as pd


class LinearRegression(object):

    # setting learning rate and iteration times
    def __init__(self, alpha=0.0005, lamb=0.1, iters=100):
        self.iters = iters
        self.alpha = alpha
        self.lamb = lamb
        # add one line for intercept
        self.theta = np.array([0.0] * (X.shape[1] + 1))

    def __normalize(self, tensor):
        return (tensor - np.mean(tensor, axis=0)) / np.std(tensor, axis=0)

    def __addIntercept(self, tensor):
        intercept = np.ones((tensor.shape[0], 1))
        return np.append(intercept, tensor, axis=1)

    def __preprocess(self, tensor):
        # feature normalize
        tensor = self.__normalize(tensor)
        # add constant column to x(to accommodate the θ0 intercept term)
        tensor = self.__addIntercept(tensor)
        return tensor

    def __hypotheses(self, X, theta):
        return np.dot(X, theta)

    def computeCost(self, X, y, theta):
        inner_prodect = np.power(
            self.__hypotheses(X, theta) - y,
            2
        )
        return np.sum(inner_prodect) / (2 * X.shape[0]) + self.lamb * np.dot(theta[1:], theta[1:]) / 2 * X.shape[0]

    def __gradientDescent(self, X, y, theta):
        for i in range(self.iters):
            # use vectorization implementation to optimize performance

            # do not penalize θ0.
            theta[0:1] = theta[0:1] - (self.alpha / X.shape[0]) * (
                np.dot(X[:, 0:1].T, (self.__hypotheses(X[:, 0:1], theta[0:1]) - y)))
            theta[1:] = theta[1:] - (self.alpha / X.shape[0]) * (np.dot(
                X[:, 1:].T, (self.__hypotheses(X[:, 1:], theta[1:]) - y)) + self.lamb * theta[1:])

            print("the cost of iteration {} is {}".format(
                i, self.computeCost(X, y, theta)))

        return theta

    def fit(self, X, y):
        X = self.__preprocess(X)
        self.theta = self.__gradientDescent(X, y, self.theta)

    def predict(self, X):
        X = self.__preprocess(X)
        return self.__hypotheses(X, self.theta)


if __name__ == "__main__":
    import random
    # generate feature X
    X = np.arange(0, 20).reshape(10, 2)

    # generate sample response
    y = np.arange(
        0, 10) + np.array([random.random() for r in range(0, 10)])

    lr = LinearRegression()
    lr.fit(X, y)
    print(lr.predict(X))
