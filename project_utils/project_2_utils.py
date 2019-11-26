import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"

def load_data():
    
    file_path = tf.keras.utils.get_file("wdbc_data", DATA_URL)

    df = pd.read_csv(file_path, header = None)
    df = df.sample(frac=1, replace=True, random_state=1)
    
    y = df[1].values
    x = df.drop([0, 1], axis = 1).values
    
    y = np.array([int(s == 'M') for s in y])
    y = np.reshape(y, (-1,1))
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify = y, random_state=42)

    scaler = MinMaxScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, x_test, y_train, y_test