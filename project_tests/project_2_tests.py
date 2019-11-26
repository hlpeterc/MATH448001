import pandas as pd
import numpy as np

import sys


def test_initialize_weights_and_biases(function):

    function_inputs = [5,4,4,2]

    function_correct_outputs = {'biases': [np.array([[0., 0., 0., 0.]]),
                                           np.array([[0., 0., 0., 0.]]),
                                           np.array([[0., 0.]])],
                                'weights': [np.array([[-0.1355025 ,  0.35978839, -0.81630981, -0.32278956],
                                                   [-0.57684521, -0.66570829, -0.51233493, -0.25219828],
                                                   [-0.16857801,  0.06338746, -0.13195481,  0.30246218],
                                                   [-0.48262746,  0.61746319, -0.77177283,  0.27837228],
                                                   [-0.13504069,  0.09584009, -0.58724567, -0.4929982 ]]),
                                            np.array([[ 0.52090487,  0.81105284, -0.3231588 ,  0.33311254],
                                                   [ 0.65192514,  0.68347879, -0.71872451, -0.79838053],
                                                   [-0.57187049,  0.65496203, -0.69568369, -0.1366456 ],
                                                   [ 0.79308793,  0.05744396,  0.33234091, -0.3195363 ]]),
                                            np.array([[ 0.37300186,  0.66925134],
                                                   [-0.96342345,  0.50028863],
                                                   [ 0.97772218,  0.49633131],
                                                   [-0.43911202,  0.57855866]])]}

    function_raw_outputs = function(function_inputs)


    assert isinstance(function_raw_outputs, type(function_correct_outputs)), \
        'Wrong type for output. Got {}, expected {}'.format(type(function_raw_outputs), type(function_correct_outputs))

    if not same_keys(function_raw_outputs, function_correct_outputs):
        print('The keys of the output dictionary does not match the correct output. Expected keys are {}'.format(function_correct_output.keys()))


    if same_dictionary(function_raw_outputs, function_correct_outputs):
        print('Function is correct')
    else:
        print('For the input {}, your function output is {} \n. The correct output is {}'. format(function_inputs,
                                                                                                  function_raw_outputs,
                                                                                                  function_correct_outputs))

def same_keys(d1, d2):
    keys1 = set(d1.keys())
    keys2 = set(d2.keys())

    if keys1 - keys2:
        return False
    else:
        return True

def same_dictionary(d1, d2):
    keys = list(d1.keys())

    for key in keys:
        for arr1, arr2 in zip(d1[key], d2[key]):
            if not np.isclose(arr1, arr2, equal_nan = True).all():
                return False

    return True

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
            print('For x={}, the expected output is {}, your output is {}'.format(x, expected_outputs[func.__name__][i], output))
    if check:
        print('Test Passed')

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
