import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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
    plt.legend()

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
