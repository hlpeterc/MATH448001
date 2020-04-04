import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds


FASHION_MNIST_CLASS_NAMES = [ 'T-shirt/top', 'Trouser',  'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',  'Ankle boot']

def load_fashion_mnist_data():

    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.fashion_mnist.load_data()

    x_train = x_train/255.
    x_val = x_val/255.

    return np.expand_dims(x_train, axis = -1), y_train, np.expand_dims(x_val, axis = -1), y_val

def plot_fashion_mnist_data(images, labels):

    images = images * 255.
    n = len(images)

    plt.figure(figsize=(10*n,20))
    for i in range(n):
        plt.subplot(n,1,i+1)
        plt.imshow(images[i].reshape((28,28)),  cmap=plt.cm.gray)
        plt.title(FASHION_MNIST_CLASS_NAMES[labels[i]])


def load_cats_dogs():
    dataset = tfds.load('cats_vs_dogs', split = 'train', shuffle_files=True)

    def resize(item):
        return tf.image.resize(item['image'], (128,128)), item['label']

    dataset = dataset.map(resize)

    x_train = []
    y_train = []
    for item in tfds.as_numpy(dataset):
        x_train.append(np.expand_dims(item[0], axis = 0))
        y_train.append(item[1])

    x_train = np.concatenate(x_train, axis = 0)
    y_train = np.array(y_train).reshape(-1,1)

    x_train = x_train / 127.5 - 1
    return x_train[0:20000], y_train[0:20000], x_train[20000:], y_train[20000:]
