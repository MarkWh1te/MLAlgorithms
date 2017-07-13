# -*- coding: utf-8 -*-
###############################################################################
#                     mark's demo code for backpropagation                    #
###############################################################################
import numpy as np


def sigmoid(x):
    return 1.0/(1.0 + np.exp(x))

# the derived function of sigmoid function
def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

class NeuralNetwork(object):

    def __init__(self,inputs,hidden,outputs):
        """
        :param inputs: number of input laryer neurons
        :param hiddens: number of hidden laryer neurons
        :param outputs: number of output laryer neurons
        """
        self.inputs = inputs + 1 # add one for bias node
        self.hidden = hidden
        self.outputs = outputs
        # set up array for activation
        self.inputs_array = [1.0] * self.inputs
        self.hidden_array = [1.0] * self.hidden
        self.outputs_array = [1.0] * self.outputs
        # generate random weight for start
        self.weight1 = np.random.randn(self.inputs,self.hidden)
        self.weight2 = np.random.randn(self.hidden,self.outputs)
        # create arrays of 0 for changes
        self.change_inputs = np.zeros((self.inputs,self.hidden))
        self.change_outputs = np.zeros((self.hidden,self.outputs))

    def feedForword(self,inputs_data):

        if len(inputs_data) != self.inputs -1:
            raise ValueError('Wrong number of inputs!')

        # inputs activation
        for i in range(self.inputs-1):
            self.inputs_array = inputs_data[i]

        # hidden activation
        for j in range(self.hidden):
            sum_num = 0.0
            for i in range(self.inputs):
                sum_num += self.inputs_array[i] * self.weight1[i][j]
            self.hidden_array = sigmoid(sum_num)

        # outputs activation
        for k in range(self.output):
            sum_num = 0.0
            for j in range(self.hidden):
                sum_num += self.hidden_array[j] * self.weight2[j][k]
            self.outputs_array[k] = sigmoid(sum_num)
        return self.outputs_array

    def backPropagate(self,target,N):
        """
        :param targets: y values
        :param N: learning rate
        :return: updated weights and current error
        """
        if len(targets) != self.outputs:
            raise ValueError('targets number not right')






if __name__ == '__main__':
    pass
