import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



def load_partial_mnist():
    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.mnist.load_data(path='mnist.npz')

    x_train = x_train[(y_train == 3) + (y_train == 5) + (y_train == 6) + (y_train == 8)]
    y_train = y_train[(y_train == 3) + (y_train == 5) + (y_train == 6) + (y_train == 8)]
    x_val = x_val[(y_val == 3) + (y_val == 5) + (y_val == 6) + (y_val == 8)]
    y_val = y_val[(y_val == 3) + (y_val == 5) + (y_val == 6) + (y_val == 8)]

    y_train[y_train == 3] = 0
    y_val[y_val == 3] = 0
    y_train[y_train == 5] = 1
    y_val[y_val == 5] = 1
    y_train[y_train == 6] = 2
    y_val[y_val == 6] = 2
    y_train[y_train == 8] = 3
    y_val[y_val == 8] = 3

    x_train = x_train[:2000].reshape((-1, 28,28,1))
    y_train = y_train[:2000]
    x_val = x_val.reshape((-1, 28,28,1))

    def augment_image(image):
        image = tf.keras.preprocessing.image.random_shift(image, 0.2, 0.2)
        image = tf.keras.preprocessing.image.random_rotation(image, 90, row_axis=0, col_axis=1, channel_axis=2)
        return image

    for i in range(len(x_train)):
        x_train[i] = augment_image(x_train[i])

    for i in range(len(x_val)):
        x_val[i] = augment_image(x_val[i])

    return x_train/255., y_train, x_val/255., y_val


def plot_sample_data(x_train):
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[i].reshape((28,28)), cmap=plt.cm.gray)
        plt.xlabel(y_train[i])
    plt.show()

def plot_learning_curve(history):
    d = history.history
    df = pd.DataFrame(d)
    key_list = list(d.keys())

    if len(key_list) == 4:
        df[[key_list[0], key_list[2]]].plot()
        df[[key_list[1], key_list[3]]].plot()

    if len(key_list) == 2:
        df[[key_list[0], key_list[1]]].plot()

def load_regression_data():

    np.random.seed(42)
    x = np.random.uniform(-1,1, size = (5000, 2))
    x_val = np.random.uniform(-1,1, size = (1000, 2))

    y = 5*np.sin(2*np.pi*x[:,0])*np.sin(2*np.pi*x[:,1]) + np.random.randn(len(x))*abs(x[:,0])/10
    y_val = 5*np.sin(2*np.pi*x_val[:,0])*np.sin(2*np.pi*x_val[:,1]) + np.random.randn(len(x_val))*abs(x_val[:,0])/10

    return x, y.reshape((-1,1)), x_val, y_val.reshape((-1,1))


def plot_sample_data_2(x,y):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    xs = x[:,0]
    ys = x[:,1]
    zs = y
    ax.scatter(xs, ys, zs, c=zs)

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$y$')
    plt.show()

def average_metric(history):
    d = history.history

    key_list = list(d.keys())

    val_loss_average = np.mean(d[key_list[1]][-5:])

    return val_loss_average

def plot_principal_components(X, principal_components):
    v = principal_components
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, c=z)

    origin = 3 * [[0, 0, 0]]

    ax.quiver(*origin, v[0,:], v[1, :], v[2,:], color = 'r', length = 20)

    ax.view_init(5,5)

    fig.show()

def plot_2D_projection(X_2D, v):
    Xp = X_2D[:,0].reshape(-1,1)*v[:,0].reshape(1,-1) + X_2D[:,1].reshape(-1,1)*v[:,1].reshape(1,-1)

    x = Xp[:,0]
    y = Xp[:,1]
    z = Xp[:,2]

    fig = plt.figure();
    ax = fig.add_subplot(111, projection='3d');
    ax.scatter(x,y,z, c=z);

    origin = 3 * [[0, 0, 0]]

    ax.quiver(*origin, v[0,:], v[1, :], v[2,:], color = 'r', length = 15);

    ax.view_init(0, 5);

def plot_1D_projection(X_1D, v):
    # Plot the projection onto the first principal component in the original space
    Xp = X_1D.reshape(-1,1)*v[:,0].reshape(1,-1)

    x = Xp[:,0]
    y = Xp[:,1]
    z = Xp[:,2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x,y,z, c=z)

    origin = 3 * [[0, 0, 0]]

    ax.quiver(*origin, v[0,:], v[1, :], v[2,:], color = 'r', length = 15)

    ax.view_init(5, 5);
