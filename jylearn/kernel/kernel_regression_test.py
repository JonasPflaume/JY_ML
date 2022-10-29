import torch as th
import torch.nn as nn
from kernels import RBF, Constant, DotProduct
import numpy as np

device = "cuda" if th.cuda.is_available() else "cpu"

import matplotlib.pyplot as plt
l = np.ones(1,) * 0.1 
train_data_num = 10 # play around those parameters
rbf = RBF(1., l, 1)

# kernel regression
kernel = rbf
X = th.linspace(-5,5,100).reshape(-1,1).to(device).double()
Y = th.cos(X)
Xtrain = th.linspace(-5,5,train_data_num).reshape(-1,1).to(device).double()
Ytrain = th.cos(Xtrain) + th.randn(train_data_num,1).to(device).double() * 0.2
import time
s = time.time()
for i in range(1000):
    temp = kernel(X, Xtrain)
    pred = temp @ Ytrain / th.sum(temp, dim=1, keepdim=True)
e = time.time()
print("The time for each pred: %.5f" % ((e-s)/1000))
plt.plot(X.detach().cpu().numpy(), pred.detach().cpu().numpy(), label="Prediction")
plt.plot(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), label="GroundTueth")
plt.plot(Xtrain.detach().cpu().numpy(), Ytrain.detach().cpu().numpy(), 'rx', label="data", alpha=0.2)
plt.grid()
plt.legend()
plt.show()