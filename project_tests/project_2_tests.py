import pandas as pd
import numpy as np

import sys

GREEN = '\033[32m'
RED = '\033[31m'
BLACK = '\033[30m'

def test_initialize_weights_and_biases(func):

    inputs = [12, 2]

    expected_output_W, expected_output_b = np.array([[-0.1086437 ,  0.28847248],
                                [-0.65450392, -0.25880741],
                                [-0.46250511, -0.53375407],
                                [-0.41078181, -0.20220847],
                                [-0.1351631 ,  0.05082303],
                                [-0.10579922,  0.24250925],
                                [-0.38696284,  0.49507194],
                                [-0.61879489,  0.22319436],
                                [-0.10827343,  0.07684302],
                                [-0.47084402, -0.39527794],
                                [ 0.39376707,  0.61309832],
                                [-0.24428509,  0.25180941]]), np.array([[0., 0.]])


    output_W, output_b = func(*inputs)

    if np.isclose(expected_output_W, output_W).all() and np.isclose(expected_output_b, output_b).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the inputs: {} \nThe expected output {}, {} \nYour function output is {}, {}'.format(
                inputs, expected_output_W, expected_output_b, output_W, output_b))



def test_activations(func):

    x_in = [np.array([0.30233257263183977]),
            np.array([0.18626021, 0.34556073]),
            np.array([-0.10961306, -0.05189851,  0.55074457,  0.71826158,  0.06342418])
           ]


    expected_outputs = {'sigmoid': [np.array([0.57501263]),
                                    np.array([0.5464309 , 0.58554065]),
                                    np.array([0.47262414, 0.48702828, 0.63430832, 0.67222409, 0.51585073])],
                        'sigmoid_derivative' : [np.array([0.2443731]),
                                                np.array([0.24784417, 0.2426828 ]),
                                                np.array([0.24925056, 0.24983173, 0.23196128, 0.22033886, 0.24974875])],
                        'relu': [np.array([0.30233257]),
                                 np.array([0.18626021, 0.34556073]),
                                 np.array([0.        , 0.        , 0.55074457, 0.71826158, 0.06342418])],
                        'relu_derivative': [np.array([1.]),
                                            np.array([1., 1.]),
                                            np.array([0., 0., 1., 1., 1.])],
                        'softmax': [np.array([1.]),
                                    np.array([0.46025888, 0.53974112]),
                                    np.array([0.13382837, 0.14177946, 0.25902273, 0.30625951, 0.15910994])]
                       }
    check = True
    for i, x in enumerate(x_in):
        output = func(x)
        if not np.isclose(expected_outputs[func.__name__][i], output).all():
            check = False
            print(RED + 'Test did not passed' + BLACK)
            print('For x={}, the expected output is {}, your output is {}'.format(x, expected_outputs[func.__name__][i], output))
    if check:
        print(GREEN + 'Test Passed' + BLACK)

def test_loss(func):

    np.random.seed(seed = 1)
    y = np.random.randint(0,2,50).reshape(-1,1)
    y_hat = np.random.random(50).reshape(-1,1)

    expected_output = 0.8862180846119079

    output = func(y_hat, y)

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For y_hat = {} and y = {}, the expected output is {}, your output is {}'.format(y_hat, y, expected_output, output))


def test_forward_pass(func):
    np.random.seed(seed = 1)

    inputs = np.random.randn(4,5)

    np.random.seed(seed = 1)
    weights = np.random.uniform(low = -1, high = 1, size = (5, 1))
    biases = np.zeros((1, 1))

    expected_output = np.array([[0.45058505],
                             [0.87673696],
                             [0.18584077],
                             [0.63547328]])

    output = func(inputs, weights, biases)

    if np.isclose(expected_output, output).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the input = {}, \n weights = {} and, \n biases = {}, the expected output is {}, your output is {}'.format(inputs, weights, biases, expected_output, output))


def test_crossentropy_loss(func):
    np.random.seed(seed = 1)
    y = np.random.randint(0,2,4)
    y_hat = np.random.random(4)

    expected_outputs = np.array([-9.07602963, -1.19622763, -0.1587096 , -0.09688387])

    check = True
    for i in range(4):
        output = func(y_hat[i], y[i])
        if not np.isclose(expected_outputs[i], output):
            check = False
            print(RED + 'Test did not passed' + BLACK)
            print('For y_hat={} and y = {}, the expected output is {}, your output is {}'.format(y_hat[i], y[i], expected_outputs[i], output))
    if check:
        print(GREEN + 'Test Passed' + BLACK)


def test_update_parameters(func):

    np.random.seed(seed = 1)

    x = np.random.randn(4,5)
    y = np.random.randint(0,2,5)

    np.random.seed(seed = 1)
    weights = np.random.uniform(low = -1, high = 1, size = (5, 1))
    biases = np.zeros((1, 1))

    learning_rate = 0.01

    expected_outputs = update_parameters(x, y, weights, biases, learning_rate)


    check = True
    outputs = func(x,y,weights,biases,learning_rate)
    for i in range(2):
        if not np.isclose(expected_outputs[i], outputs[i]).all():
            check = False
            print(RED + 'Test did not passed' + BLACK)
            print('For x={}, y = {}, weights = {}, biases = {} and learning_rate = {}, the expected outputs are {}, your outputs are {}'.format(x, y, weights, biases, learning_rate, expected_outputs, outputs))
            break
    if check:
        print(GREEN + 'Test Passed' + BLACK)


def test_accuracy(func):
    input_y = np.array([[0],[0],[1],[1]])
    input_y_hat = np.array([[0.3],[0.7],[0.4],[0.8]])

    expected_output = 0.5

    output = func(input_y_hat, input_y)

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For y_hat = {} and y = {}, the expected output is {}, your output is {}'.format(input_y_hat, input_y, expected_output, output))
