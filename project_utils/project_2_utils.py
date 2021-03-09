import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

def load_data(n_features = 2):

    file_path = tf.keras.utils.get_file("wdbc_data", DATA_URL)

    df = pd.read_csv(file_path, header = None)

    y = df[1].values
    x = df.drop([0, 1], axis = 1).values

    y = np.array([int(s == 'M') for s in y])
    y = np.reshape(y, (-1,1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify = y, random_state=2)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)



    return x_train[:,:n_features], x_test[:,:n_features], y_train, y_test

def plot_data(x,y):
    plt.scatter(x[:,0], x[:,1], c=y.squeeze())

def plot_decision_boundary(weights, biases, x, y):
    w1 = weights[0,0]
    w2 = weights[1,0]
    b = biases[0,0]

    #y_hat = forward_pass(x, weights, biases)
    #pred = y_hat >= 0.5
    #mask = np.array(pred != y).squeeze()

    t = np.linspace(0.1,0.6,100)
    t_hat = (-b-t*w1)/w2

    plt.plot(t,t_hat)
    plt.scatter(x[:,0], x[:,1], c=y.squeeze())



# Linear regression utility functions

def plot_regression_line(w, b):
    t = np.linspace(-0.1, 1.1, 2000)
    plt.plot(t, w*t + b, 'b-')

def load_regression_data():
    np.random.seed(seed = 1)
    x_train = np.random.random([100])
    y_train = 5*x_train - 3 + np.random.randn(100)

    x_test = np.random.random([100])
    y_test = 5*x_test - 3 + np.random.randn(100)

    x_train = x_train.reshape(-1,1)
    x_test = x_test.reshape(-1,1)

    y_train = y_train.reshape(-1,1)
    y_test = y_test.reshape(-1,1)

    return x_train, y_train, x_test, y_test

def plot_regression_data(x, y):
    plt.plot(x.squeeze(),y.squeeze(), '.')
    plt.xlabel('x')
    plt.ylabel('y')

def plot_regression_lines(weight_bias_memory, x, y):
    for i, (weight, bias) in enumerate(weight_bias_memory):
        if i % 10 == 0 or i == len(weight_bias_memory) - 1:
            plot_regression_line(weight, bias)
    plt.plot(x[:, -1], y, 'o')

def mse_from_weights(w, b, x, y):
    y_hat = np.dot(x, w) + b
    return np.mean((y-y_hat)**2)

def plot_gradient_descent_progression(weight_bias_memory, x_train, y_train):
    delta = 0.1
    w = np.arange(-2.0, 10.0, delta)
    b = np.arange(-8.0, 2.0, delta)
    W, B = np.meshgrid(w,b)
    mse = np.zeros((len(b),len(w)))

    for i in range(len(w)):
        for j in range(len(b)):
            mse[j,i] = mse_from_weights(w[i],b[j], x_train, y_train)

    weight_bias_memory = np.array(weight_bias_memory)
    fig, ax = plt.subplots()
    CS = ax.contour(W, B, mse, levels = 200)
    ax.plot(weight_bias_memory[:,0], weight_bias_memory[:,1], 'ro')
    plt.xlabel('w');
    plt.ylabel('b');

def load_auto_mpg_data():
    dataset_path = tf.keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
    dataset = pd.read_csv(dataset_path, names=column_names,
                        na_values = "?", comment='\t',
                        sep=" ", skipinitialspace=True)
    dataset = dataset.dropna()
    dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))
    dataset = pd.get_dummies(dataset)

    train_dataset = dataset.sample(frac=0.8,random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    x_train = train_dataset.values
    x_test = test_dataset.values

    y_train = train_labels.values.reshape(-1,1)
    y_test = test_labels.values.reshape(-1,1)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return train_dataset, x_train, y_train, x_test, y_test
