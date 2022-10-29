import torch as th
import torch.nn as nn
from kernels import RBF, Constant, DotProduct
import numpy as np
device = "cuda" if th.cuda.is_available() else "cpu"

#
test_case = 4

import matplotlib.pyplot as plt
l = np.ones(1,) * 0.7
rbf = RBF(1., l)
cons = Constant(0.9)
dot = DotProduct(0.1)

if test_case == 1:
    kernel = (rbf + cons + dot) ** 2.
    x = np.linspace(-7, 7, 1000)

    x = th.from_numpy(x).to(device)
    pred = kernel(x.reshape(-1,1), x.reshape(-1,1))
    pred += th.eye(1000).to(device) * 1e-8
    mean = th.zeros(1000,).to(device)
    dis = th.distributions.multivariate_normal.MultivariateNormal(mean.double(), pred.double())
    samples = [dis.sample() for _ in range(5)]
    
    for sample in samples:
        plt.plot(x.detach().cpu().numpy(), sample.detach().cpu().numpy())
    plt.grid()
    plt.show()
    
elif test_case == 2:
    kernel1 = cons + rbf ** 2. + dot
    kernel2 = rbf
    kernel3 = dot
    X = th.tensor([1.]).reshape(1,1).to(device)
    Y = th.tensor([2.]).reshape(1,1).to(device)
    res1 = kernel1(X, Y)
    res2 = kernel2(X, Y)
    res3 = kernel3(X, Y)
    
    assert th.isclose(res1, res3 + res2 ** 2. + 0.9)
    
elif test_case == 3:
    # kernel regression
    kernel = rbf + cons + dot
    X = th.linspace(-5,5,100).reshape(-1,1).to(device).double()
    Y = th.cos(X)
    Xtrain = th.linspace(-5,5,10).reshape(-1,1).to(device).double()
    Ytrain = th.cos(Xtrain) + th.randn(10,1).to(device).double() * 0.2
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
    
elif test_case == 4:
    l = np.ones(2,) * 0.7
    rbf = RBF(1., l)
    kernel = rbf ** 2.
    X = th.tensor([1., 1.]).reshape(1,2).to(device)
    Y = th.tensor([2., 2.]).reshape(1,2).to(device)
    res = kernel(X, Y)
    res.backward()
    for parameter in rbf.parameters():
        print(parameter.grad)