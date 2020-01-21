import numpy as np
import matplotlib.pyplot as plt


def load_data():
    np.random.seed(seed=1)
    x_train = np.r_[np.random.randn(100,2)*0.5 + 1, np.random.randn(100,2)*0.5 + -1]
    y_train = np.r_[np.zeros(100), np.ones(100)]

    return x_train, y_train

def plot_data(x,y):
    plt.scatter(x[:,0], x[:,1], c = y)

def plot_decision_boundary(w):
    t = np.linspace(-2,2,100)
    plt.plot(t, (-1/w[1])*w[0]*t)
