import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLASS_NAMES = [ 'T-shirt/top', 'Trouser',  'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',  'Ankle boot']

def load_data():

    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train/255.
    x_val = x_val/255.

    return x_train.reshape((-1, 784)), y_train, x_val.reshape((-1, 784)), y_val

def plot_data(images, labels):

    images = images * 255.
    n = len(images)

    plt.figure(figsize=(10*n,15))
    for i in range(n):
        plt.subplot(n,1,i+1)
        plt.imshow(images[i].reshape((28,28)),  cmap=plt.cm.gray)
        plt.title(CLASS_NAMES[labels[i]])
