# -*- coding: utf-8 -*-
import numpy as np


def sigmoid(x):
    return 1.0/(1.0 + np.exp(x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x ** 2


class NeuralNetwork(object):

    def __init__(self):
        pass


if __name__ == '__main__':
    print(sigmoid(10))
