# -*- coding: utf-8 -*-
from random import choice
from numpy import array,dot,random



unit_step = lambda x:0 if x < 0 else 1

def train(training_data,errors,eta,n,weight):
    for i in range(n):
        x,expect = choice(training_data)
        output = dot(x,weight)
        error = expect - unit_step(output)
        errors.append(error)
        weight += eta * error * x
    return weight,errors


def show(training_data,weight):
    for x, _ in training_data:
      results = dot(x,weight)
      print("{}:{} -> {}".format(x[:2],results,unit_step(results)))


def perceptron(training_data):
    # constant variable declare
    eta = 0.1
    n = 100
    weight = random.rand(3)
    errors = []
    weight ,errors = train(training_data,errors,eta,n,weight)
    show(training_data,weight)


if __name__ == '__main__':
    training_data_or = [
        (array([0,0,1]), 0),
        (array([0,1,1]), 1),
        (array([1,0,1]), 1),
        (array([1,1,1]), 1),
    ]

    training_data_and = [
        (array([0,0,1]), 0),
        (array([0,1,1]), 0),
        (array([1,0,1]), 0),
        (array([1,1,1]), 1),
    ]

    training_data_not = [
        (array([0,0,1]), 1),
        (array([0,1,1]), 0),
        (array([1,0,1]), 0),
        (array([1,1,1]), 1),
    ]
    # data_list = [training_data_not,training_data_and,training_data_or]
    data_list = [training_data_and,training_data_or]
    map(perceptron,data_list)
