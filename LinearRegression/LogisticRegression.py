import numpy as np

class LogisticRegression(object):
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

    def __sigmoid(self,tensor):
        return 1/(np.exp(-tensor)+1)

    def __hypothese(self,X,theta):
        return self.__sigmoid(np.dot(
            X,theta
            ))

    def computeCost(self,X,y,theta):
        positive = np.multiply((-y),np.log(self.__hypothese(X,theta)))
        negative = np.multiply((1-y),(1-np.log(self.__hypothese(X,theta))))
        penalty  = self.lamb/( 2*(X.shape[0]) ) * np.dot(theta[1:],theta[1:])
        return np.sum(positive-negative)/len(X)

    def __gradientDescent(self,X,y,theta):
        for i in range(self.iters):
            # use vectorization implementation to optimize performance

            # do not penalize θ0.
            theta[0:1] = theta[0:1] - (self.alpha/X.shape[0])*(
                np.dot(X[:,0:1].T,self.__hypothese(X[:,0:1],theta[0:1])-y)
            )


            # calculate others theta with regulation
            theta[1:] = theta[0:1] - (self.alpha/X.shape[0])*(
                np.dot(X[:,1:].T,self.__hypothese(X[:,1:],theta[1:])-y)
             + self.lamb * theta[1:])

            print("the cost of iteration {} is {}".format(
                i, self.computeCost(X, y, theta)))

        return theta

    def fit(self, X, y):
        X = self.__preprocess(X)
        print(X)
        self.theta = self.__gradientDescent(X, y, self.theta)

    def predict(self, X):
        X = self.__preprocess(X)
        pred = self.__hypothese(X,self.theta)
        numpy.putmask(pred,pred>=0.5,1.0)
        numpy.putmask(pred,pred<0.5,0.0)
        return pred



if __name__ == "__main__":
    pass
