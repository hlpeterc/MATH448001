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
        print('Test Passed')
    else:
        print('For y_hat = {} and y = {}, the expected output is {}, your output is {}'.format(y_hat, y, expected_output, output))
