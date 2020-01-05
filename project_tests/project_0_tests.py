import numpy as np

GREEN = '\033[32m'
RED = '\033[31m'
BLACK = '\033[30m'

def test_square_function(func):
    x_inputs = np.random.uniform(1,100, 10)

    expected_outputs = x_inputs**2

    if (func(x_inputs) == expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the inputs: x = {} \nThe expected outputs are {} \nYour function outputs are {}'.format(
                x_inputs, expected_outputs, func(x_inputs)))

def test_normal_pdf(func):

    inputs = ((-2, 0.1, -1.5), (-1, 0.2, -1.4), (2, 0.5, 3.5))

    expected_outputs = [1.4867195147343004e-05, 0.2699548325659406, 0.008863696823876015]

    outputs = [func(*x) for x in inputs]

    if np.isclose(outputs, expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the inputs: {} \nThe expected outputs are {} \nYour function outputs are {}'.format(
                inputs, expected_outputs, outputs))


def test_is_even(func):

    x_inputs = np.random.randint(-100, 100, 10)

    expected_outputs = x_inputs % 2 == 0

    outputs = [func(x) for x in x_inputs]

    if np.isclose(outputs, expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the inputs: {} \nThe expected outputs are {} \nYour function outputs are {}'.format(
                x_inputs, expected_outputs, outputs))


def test_sum_of_squares(func):

    x_inputs = np.random.uniform(-10, 10, size = (10, 5))

    expected_outputs = np.sum(x_inputs**2, axis = 1)

    outputs = [func(x) for x in x_inputs]

    if np.isclose(outputs, expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the inputs: {} \nThe expected outputs are {} \nYour function outputs are {}'.format(
                x_inputs, expected_outputs, outputs))

def test_taylor_exp(func):

    inputs = ((3, 50), (-1, 20), (0.5, 30))

    expected_outputs = [20.08553692318766, 0.36787944117144245, 1.6487212707001278]

    outputs = [func(*x) for x in inputs]

    if np.isclose(outputs, expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the inputs: {} \nThe expected outputs are {} \nYour function outputs are {}'.format(
                inputs, expected_outputs, outputs))

def test_is_prime(func):

    n_inputs = np.arange(2,50)

    expected_outputs = np.array([ True,  True, False,  True, False,  True, False, False, False,
        True, False,  True, False, False, False,  True, False,  True,
       False, False, False,  True, False, False, False, False, False,
        True, False,  True, False, False, False, False, False,  True,
       False, False, False,  True, False,  True, False, False, False,
        True, False, False])

    outputs = np.array([func(n) for n in n_inputs])

    if (outputs == expected_outputs).all():
        print(GREEN + 'Test passed')
    else:
        print(RED + 'Test did not passed' + BLACK)
        print('For the inputs: {} \nThe expected outputs are {} \nYour function outputs are {}'.format(
                inputs, expected_outputs, outputs))

def test_factorize(func):

    n_inputs = np.arange(2,100)

    expected_outputs = np.array([list([2]), list([3]), list([2, 2]), list([5]), list([2, 3]),
                                list([7]), list([2, 2, 2]), list([3, 3]), list([2, 5]), list([11]),
                                list([2, 2, 3]), list([13]), list([2, 7]), list([3, 5]),
                                list([2, 2, 2, 2]), list([17]), list([2, 3, 3]), list([19]),
                                list([2, 2, 5]), list([3, 7]), list([2, 11]), list([23]),
                                list([2, 2, 2, 3]), list([5, 5]), list([2, 13]), list([3, 3, 3]),
                                list([2, 2, 7]), list([29]), list([2, 3, 5]), list([31]),
                                list([2, 2, 2, 2, 2]), list([3, 11]), list([2, 17]), list([5, 7]),
                                list([2, 2, 3, 3]), list([37]), list([2, 19]), list([3, 13]),
                                list([2, 2, 2, 5]), list([41]), list([2, 3, 7]), list([43]),
                                list([2, 2, 11]), list([3, 3, 5]), list([2, 23]), list([47]),
                                list([2, 2, 2, 2, 3]), list([7, 7]), list([2, 5, 5]),
                                list([3, 17]), list([2, 2, 13]), list([53]), list([2, 3, 3, 3]),
                                list([5, 11]), list([2, 2, 2, 7]), list([3, 19]), list([2, 29]),
                                list([59]), list([2, 2, 3, 5]), list([61]), list([2, 31]),
                                list([3, 3, 7]), list([2, 2, 2, 2, 2, 2]), list([5, 13]),
                                list([2, 3, 11]), list([67]), list([2, 2, 17]), list([3, 23]),
                                list([2, 5, 7]), list([71]), list([2, 2, 2, 3, 3]), list([73]),
                                list([2, 37]), list([3, 5, 5]), list([2, 2, 19]), list([7, 11]),
                                list([2, 3, 13]), list([79]), list([2, 2, 2, 2, 5]),
                                list([3, 3, 3, 3]), list([2, 41]), list([83]), list([2, 2, 3, 7]),
                                list([5, 17]), list([2, 43]), list([3, 29]), list([2, 2, 2, 11]),
                                list([89]), list([2, 3, 3, 5]), list([7, 13]), list([2, 2, 23]),
                                list([3, 31]), list([2, 47]), list([5, 19]),
                                list([2, 2, 2, 2, 2, 3]), list([97]), list([2, 7, 7]),
                                list([3, 3, 11])], dtype=object)

    outputs = np.array([func(n) for n in n_inputs])

    check = True

    for i in range(len(n_inputs)):
        if expected_outputs[i] != sorted(outputs[i]):
            check = False
            if not check:
                print(RED + 'Test did not passed' + BLACK)
                print('For the input: {} \nThe expected output is {} \nYour function output is {}'.format(
                        n_inputs[i], expected_outputs[i], outputs[i]))
    if check:
        print(GREEN + 'Test passed')
