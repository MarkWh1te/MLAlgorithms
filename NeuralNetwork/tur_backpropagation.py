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
        self.chang_inputs = np.zeros((self.inputs,self.hidden))
        self.chang_outputs = np.zeros((self.hidden,self.outputs))





if __name__ == '__main__':
    pass
