import numpy as np
import tensorflow as tf

GREEN = '\033[32m'
RED = '\033[31m'
BLACK = '\033[30m'


def test_eigendecomposition(func):

    input_A = np.array([[1,2,3],[2,5,4],[3,4,8]])

    expected_output_P = np.array([[ 0.31251404,  0.94962   ,  0.0235971 ],
                                  [ 0.53935014, -0.15693851, -0.82732807],
                                  [ 0.78194399, -0.27127874,  0.56122317]])

    expected_output_D = np.array([[11.95801072,  0.        ,  0.        ],
                                  [ 0.        , -0.18754158,  0.        ],
                                  [ 0.        ,  0.        ,  2.22953087]])

    output_D, output_P = func(input_A)

    assert expected_output_P.shape == output_P.shape, "For the input {} the expected \
output shape is {} but got output of shape {}".format(input_A, expected_output_P.shape, output_P.shape)

    assert expected_output_D.shape == output_D.shape, "For the input {} the expected \
output shape is {} but got output of shape {}".format(input_A, expected_output_D.shape, output_D.shape)


    if np.isclose(expected_output_P, output_P).all() and np.isclose(expected_output_D, output_D).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the input: {} \nThe expected output {}, {} \nYour function output is {}, {}'.format(
                input_A, expected_output_D, expected_output_P, output_D, output_P))


def test_svd(func):

    np.random.seed(seed = 42)
    input_A = np.random.randn(4,5)

    expected_output_U = np.array([[ 0.27417936, -0.49296783,  0.79859313, -0.20989859],
                                  [ 0.13113305,  0.81939406,  0.5163841 ,  0.21153004],
                                  [-0.76495679,  0.15014816,  0.19924523, -0.59379961],
                                  [-0.56786522, -0.251061  ,  0.2364272 ,  0.74739524]])

    expected_output_V = np.array([[ 0.22676793, -0.17301671,  0.03873517, -0.81044709,  0.51021234],
                                  [ 0.32026729,  0.73281909,  0.28598979, -0.31826285, -0.42109814],
                                  [-0.02482828,  0.12650776,  0.79476804,  0.31884283,  0.50006235],
                                  [ 0.67954157, -0.56615737,  0.28994071,  0.10430211, -0.35034919],
                                  [ 0.61936418,  0.31064937, -0.44831638,  0.3596451 ,  0.43537663]])

    expected_output_sv = np.array([3.4364811 , 2.11035904, 1.30357173, 0.36875989])

    output_U, output_V, output_sv = func(input_A)

    assert expected_output_U.shape == output_U.shape, "For the input {} the expected \
output shape is {} but got output of shape {}".format(input_A, expected_output_U.shape, output_U.shape)

    assert expected_output_V.shape == output_V.shape, "For the input {} the expected \
output shape is {} but got output of shape {}".format(input_A, expected_output_V.shape, output_V.shape)


    if np.isclose(expected_output_U, output_U).all() and np.isclose(expected_output_V, output_V).all() and \
np.isclose(expected_output_sv, output_sv).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not pass' + BLACK)
        print('For the input: {} \nThe expected output {}, {}, {} \nYour function output is {}, {}, {}'.format(
                input_A, expected_output_U, expected_output_V, expected_output_sv, output_U, output_V, output_sv))
