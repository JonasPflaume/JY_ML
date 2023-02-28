import numpy as np
from scipy.io import loadmat
import os

path = os.path.dirname(__file__)
np.random.seed(8)

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

def classification_toy_data():
    ''' 3 class artificial data
    '''
    class_1_pos = np.array([[0,0]])
    class_2_pos = np.array([[2,2],[-2,-2]])
    class_3_pos = np.array([[-2,2],[2,-2]])
    
    data_len = 1000
    class_1_X = np.random.normal(loc=class_1_pos[0], scale=np.array([1,1]), size=(data_len,2))
    
    class_2_X1 = np.random.normal(loc=class_2_pos[0], scale=np.array([0.5,1]), size=(data_len//2,2))
    class_2_X2 = np.random.normal(loc=class_2_pos[1], scale=np.array([1,0.5]), size=(data_len//2,2))
    
    class_3_X1 = np.random.normal(loc=class_3_pos[0], scale=np.array([1,1]), size=(data_len//2,2))
    class_3_X2 = np.random.normal(loc=class_3_pos[1], scale=np.array([0.5,0.5]), size=(data_len//2,2))
    
    X = np.concatenate([class_1_X, class_2_X1, class_2_X2, class_3_X1, class_3_X2])
    Y1 = np.zeros([data_len,3], dtype=int)
    Y1[:,0] = 1
    Y2 = np.zeros([data_len,3], dtype=int)
    Y2[:,1] = 1
    Y3 = np.zeros([data_len,3], dtype=int)
    Y3[:,2] = 1
    Y = np.concatenate([Y1, Y2, Y3])
    shuffle_index = np.arange(0,data_len*3)
    np.random.shuffle(shuffle_index)
    
    X = X[shuffle_index]
    Y = Y[shuffle_index]
    
    X_train, X_test, Y_train, Y_test = X[:data_len*2], X[data_len*2:], Y[:data_len*2], Y[data_len*2:]
    
    return X_train, Y_train, X_test, Y_test

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