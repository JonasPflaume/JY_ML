import numpy as np
from scipy.io import loadmat
import os

path = os.path.dirname(__file__)
np.random.seed(6)

def polynomial_data(p=12, scale=3.5):
    X = np.linspace(-1.2,1.2,200)
    weight = np.random.uniform([-15]*p,[15]*p)
    Y = 0.
    for i in range(p):
        Y += weight[i] * X ** i
    Y += np.random.normal(scale=scale, size=[200])
    return X.reshape(-1,1), Y.reshape(-1,1)
    
def robot_inv_data():
    train_data = loadmat(os.path.join(path, 'sarcos_inv.mat'))['sarcos_inv'].astype(np.float32)
    val_data, train_data = train_data[:4448], train_data[4484:].astype(np.float32)
    test_data = loadmat(os.path.join(path, 'sarcos_inv_test.mat'))['sarcos_inv_test'].astype(np.float32)

    X_train, Y_train = train_data[:, :21], train_data[:, 21:]
    X_val, Y_val = val_data[:, :21], val_data[:, 21:]
    X_test, Y_test = test_data[:, :21], test_data[:, 21:]
    
    return X_train, Y_train, X_val, Y_val, X_test, Y_test

def minist_data():
    """ without label
    """
    f = open(os.path.join(path, 'train-images.idx3-ubyte'),'rb')
    f.read(16)
    image_size = 28
    num_images = 60000
    
    buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    return data