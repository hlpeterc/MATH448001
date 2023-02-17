import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_decision_boundary(model):
    w, b = model.trainable_variables

    w = w.numpy()
    b = b.numpy()

    t = np.linspace(-0.1,1,100)
    plt.plot(t, -1/w[1]*(w[0]*t + b))

def plot_data(x,y):
    plt.scatter(x[:,0], x[:,1], c=y.squeeze())
