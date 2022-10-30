from lib2to3.pgen2.literals import simple_escapes
import torch as th
import torch.nn as nn
from torch.nn import MSELoss
from kernels import RBF, Constant, DotProduct, RQK, Matern
import numpy as np
th.manual_seed(0)
import matplotlib.pyplot as plt
device = "cuda" if th.cuda.is_available() else "cpu"
Loss = MSELoss()

l = np.ones(1,) * 0.5
train_data_num = 100 # play around those parameters
kernel = RBF(l=l, sigma=1., dim=1)
# kernel = Matern(l=l, sigma=1., mu=0.5, dim=1)

# kernel regression
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
print("MSE: %.5f" % Loss(pred, Y))
plt.plot(X.detach().cpu().numpy(), pred.detach().cpu().numpy(), label="Prediction")
plt.plot(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), label="GroundTueth")
plt.plot(Xtrain.detach().cpu().numpy(), Ytrain.detach().cpu().numpy(), 'rx', label="data", alpha=0.3)
plt.grid()
plt.legend()
plt.show()