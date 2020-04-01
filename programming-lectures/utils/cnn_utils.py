import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

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
