import numpy as np
import tensorflow as tf

GREEN = '\033[32m'
RED = '\033[31m'
BLACK = '\033[30m'

def test_column_means(func):

    inputs = np.random.uniform(-20,20, size = (10,6))

    expected_output = np.mean(inputs, axis = 0)

    output = func(inputs)

    assert expected_output.shape == output.shape, "For the input {} the expected \
output shape is {} but got output of shape {}".format(inputs, expected_output.shape, output.shape)

    if np.isclose(expected_output, output).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the input: {} \nThe expected outpu is {} \nYour function output is {}'.format(
                inputs, expected_output, output))

def test_cutoff(func):

    x_in = [(np.array([[0.28897937, 0.43569014, 0.36364435, 0.39218248],
             [0.66244581, 0.37877839, 0.3410136,  0.01200013],
             [0.62660962, 0.88664802, 0.39039634, 0.29926516]]), 0.5),
           (np.array([[0.23448093, 0.05224714],
                   [0.20100718, 0.86350585],
                   [0.55008888, 0.4784401 ],
                   [0.97934747, 0.15784187],
                   [0.55201495, 0.47493771]]), 0.6)
           ]


    expected_outputs = [np.array([[0.28897937, 0.43569014, 0.36364435, 0.39218248],
                                   [0.5       , 0.37877839, 0.3410136 , 0.01200013],
                                   [0.5       , 0.5       , 0.39039634, 0.29926516]]),
                        np.array([[0.23448093, 0.05224714],
                                   [0.20100718, 0.6       ],
                                   [0.55008888, 0.4784401 ],
                                   [0.6       , 0.15784187],
                                   [0.55201495, 0.47493771]])
                       ]

    check = True
    for i, (x, th) in enumerate(x_in):
        output = func(x, th)
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not passed' + BLACK)
            print('For the input array {} and threshold {}, the expected output is {}, your output is {}'.format(x, th, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test Passed')

def test_max_index(func):

    x_in = [np.array([[0.59924692, 0.28694015, 0.16983537, 0.89984641],
                       [0.03111706, 0.42997205, 0.54268424, 0.64534421],
                       [0.16471448, 0.75338066, 0.54912713, 0.70684244]]),
            np.array([[0.03367516, 0.45768455, 0.46902002, 0.24923532],
                       [0.21920534, 0.14771333, 0.24763095, 0.35050109],
                       [0.11549889, 0.65836753, 0.53742414, 0.8707693 ],
                       [0.8828907 , 0.06985487, 0.21786648, 0.651936  ],
                       [0.09754203, 0.58064407, 0.82283824, 0.08302386],
                       [0.70813905, 0.37341553, 0.0143709 , 0.85784191]])
           ]

    expected_outputs = [np.array([3, 3, 1]),
                       np.array([2, 3, 3, 0, 2, 3])]

    check = True
    for i, x in enumerate(x_in):
        output = func(x)
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not passed' + BLACK)
            print('For the input array {}, the expected output is {}, your output is {}'.format(x, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test Passed')

def test_linear(func):

    x_in = [(np.array([[0.46527354, 0.31002314, 0.0407845 ],
                       [0.45594427, 0.67337049, 0.51686615],
                       [0.02210991, 0.98308284, 0.5778303 ]]),
             np.array([[0.12307017, 0.79964265, 0.60164791]]),
             np.array([[0.41128018, 0.58183524, 0.89999135]])),
            (np.array([[0.88527927, 0.71755922, 0.49300952, 0.90611223],
                       [0.15506991, 0.91299081, 0.11205118, 0.85878849],
                       [0.78393805, 0.30408309, 0.21298612, 0.20433642]]),
             np.array([[0.53660877, 0.57383126, 0.10156138]]),
             np.array([[0.11599683, 0.22034654, 0.66471019, 0.02979967]]))
           ]


    expected_outputs = [np.array([[0.84643634, 1.74991534, 1.66596931]]),
                        np.array([[0.75964724, 1.16018088, 1.01519306, 1.02957981]])]



    check = True
    for i, (W,x,b) in enumerate(x_in):
        output = func(W,x,b)
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not passed' + BLACK)
            print('For W={}, x={} and b={}, the expected output is {}, your output is {}'.format(W, x, b, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test Passed')

def test_uniform_matrix(func):

    x_in = [(2,3),
            (1,3),
            (2,2)]


    expected_outputs = [np.array([[-0.16595599,  0.44064899, -0.99977125],
                                  [-0.39533485, -0.70648822, -0.81532281]]),
                        np.array([[-0.16595599,  0.44064899, -0.99977125]]),
                        np.array([[-0.16595599,  0.44064899],
                                  [-0.99977125, -0.39533485]])]

    check = True
    for i, x in enumerate(x_in):
        output = func(*x)
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not passed' + BLACK)
            print('For x={}, the expected output is {}, your output is {}'.format(x, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test Passed')

def test_orthogonal(func):

    v1 = [np.array([1,0,0]), np.array([1,2,3])]
    v2 = [np.array([0, 0.5, 0.7]), np.array([-2,5,6])]

    expected_outputs = np.array([True, False])

    check = True
    for i, (u, v) in enumerate(zip(v1, v2)):
        output = func(u, v)
        if expected_outputs[i] != output:
            check = False
            print(RED + 'Test did not passed' + BLACK)
            print('For the inputs {} and {}, the expected output is {}, your output is {}'.format(u, v, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test Passed')

def test_rotation(func):

    inputs_theta = [0.5, 1.0, 5.0]

    expected_outputs = [np.array([[ 0.87758256, -0.47942554],
                                  [ 0.47942554,  0.87758256]]),
                        np.array([[ 0.54030231, -0.84147098],
                                  [ 0.84147098,  0.54030231]]),
                        np.array([[ 0.28366219,  0.95892427],
                                  [-0.95892427,  0.28366219]])
                        ]


    check = True
    for i, theta in enumerate(inputs_theta):
        output = func(theta)
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not passed' + BLACK)
            print('For the input {}, the expected output is {}, your output is {}'.format(theta, expected_outputs[i], output))
    if check:
        print(GREEN + 'Test Passed')

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
        print(RED + 'Test did not passed' + BLACK)
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
output shape is {} but got output of shape {}".format(input_A, expected_output_P.shape, output_P.shape)

    assert expected_output_V.shape == output_V.shape, "For the input {} the expected \
output shape is {} but got output of shape {}".format(input_A, expected_output_D.shape, output_D.shape)


    if np.isclose(expected_output_U, output_U).all() and np.isclose(expected_output_V, output_V).all() and \
np.isclose(expected_output_sv, output_sv).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the input: {} \nThe expected output {}, {} \nYour function output is {}, {}'.format(
                input_A, expected_output_D, expected_output_P, output_D, output_P))

def test_coin_toss(func):

    np.random.seed(seed = 1)
    inputs = (100, 0.9)

    expected_output = np.random.binomial(1, 0.9, size = 100)

    output = func(*inputs)

    if np.isclose(expected_output, output).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the input: {} \nThe expected output {} \nYour function output is {}'.format(
                inputs, expected_output, output))


def test_die_roll(func):

    np.random.seed(seed = 1)
    m, p = 30, [1/6]*6

    expected_output = np.random.multinomial(1, p, size = m)
    output = func(m, p)

    if np.isclose(expected_output, output).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the input: {}, {} \nThe expected output {} \nYour function output is {}'.format(
                m, p, expected_output, output))


def test_expected_value(func):

    inputs_f = np.array([2,3,4,5,6,7,8,9,10,11,12])
    inputs_P = np.array([1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36])

    expected_output = 7.

    output = func(inputs_f, inputs_P)

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the input: {}, {} \nThe expected output {} \nYour function output is {}'.format(
                inputs_f, inputs_P, expected_output, output))

def test_kl_divergence(func):

    inputs_P, inputs_Q = np.array([0.1, 0.2, 0.7]), np.array([0.7, 0.21, 0.09])

    expected_output = 1.23154041755978

    output = func(inputs_P, inputs_Q)

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the input: {}, {} \nThe expected output {} \nYour function output is {}'.format(
                inputs_P, inputs_Q, expected_output, output))

def test_derivative(func):

    inputs_f = lambda x: x**2
    inputs_x = 1

    expected_output = 2

    output = func(inputs_f, inputs_x)

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the inputs: {}, {} \nThe expected output {} \nYour function output is {}'.format(
                'f(x)=x^2', inputs_x, expected_output, output))

def test_gradient_descent(func):

    f = lambda x: x**2

    g = lambda x: np.cos(x)

    functions = [f, g]

    names = ['x^2', 'cos(x)']
    N = 1000
    learning_rate = 0.01

    expected_outputs = [[-4.999298157164008e-06, 2.4992982064223448e-11], [3.141173062193505, -0.9999999119715314]]

    check = True

    for i in range(2):
        output = func(functions[i], learning_rate, N)
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not passed' + BLACK)
            print('For the function {}, learning rate 0.01 and number of iterations 1000; the expected output is {}, your output is {}'.format(names[i], expected_outputs[i], output))
    if check:
        print(GREEN + 'Test Passed' + BLACK)

def test_derivative_tf(func):

    inputs_f = lambda x: x**2
    inputs_a = 1.

    expected_output = 2.

    output = func(inputs_f, inputs_a).numpy()

    if np.isclose(expected_output, output):
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the inputs: {}, {} \nThe expected output {} \nYour function output is {}'.format(
                'f(x)=x^2', inputs_a, expected_output, output))

def test_gradient_descent_tf(func):

    f = lambda x: x**2

    g = lambda x: tf.cos(x)

    functions = [f, g]

    names = ['x^2', 'cos(x)']
    N = 1000
    learning_rate = 0.01

    expected_outputs = [[-4.9988826e-06, 2.4988827e-11], [3.135978, -0.99998426]]

    check = True

    for i in range(2):
        output_x, output_f = func(functions[i], learning_rate, N)
        output = np.c_[output_x.numpy(), output_f.numpy()]
        if not np.isclose(expected_outputs[i], output).all():
            check = False
            print(RED + 'Test did not passed' + BLACK)
            print('For the function {}, learning rate 0.01 and number of iterations 1000; the expected output is {}, your output is {}'.format(names[i], expected_outputs[i], output))
    if check:
        print(GREEN + 'Test Passed' + BLACK)
