import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

def load_dogs():
    train_dataset = tfds.load('stanford_dogs', split = 'train', shuffle_files=True)
    #test_dataset = tfds.load('stanford_dogs', split = 'test', shuffle_files=True)

    def resize(item):
        return tf.image.resize(item['image'], (160,160)), item['label']

    train_dataset = train_dataset.map(resize)
    #test_dataset = test_dataset.map(resize)

    x_train = []
    y_train = []
    #x_test = []
    #y_test = []
    for item in tfds.as_numpy(train_dataset):
        x_train.append(np.expand_dims(item[0], axis = 0))
        y_train.append(item[1])

    #for item in tfds.as_numpy(test_dataset):
    #    x_test.append(np.expand_dims(item[0], axis = 0))
    #    y_test.append(item[1])

    x_train = np.concatenate(x_train, axis = 0)
    y_train = np.array(y_train).reshape(-1,1)
    #x_test = np.concatenate(x_test, axis = 0)
    #y_test = np.array(y_test).reshape(-1,1)

    x_train = x_train / 127.5 - 1
    #x_test = x_test / 127.5 - 1

    return x_train[:10000], y_train[:10000], x_train[10000:], y_train[10000:]
