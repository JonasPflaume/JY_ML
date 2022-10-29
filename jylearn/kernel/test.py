from pickletools import read_bytes1
import torch as th
import torch.nn as nn
from kernels import RBF, Constant
import numpy as np
th.manual_seed(0)
device = "cuda" if th.cuda.is_available() else "cpu"


#### test grad computation
l = np.ones(1,) * 0.7
rbf = RBF(1., l, 1)

X = th.randn(10000,1).to(device)
Y = th.randn(10000,1).to(device)
kernel = rbf ** 2.
res = kernel(X, Y)
res = th.sum(res)
res.backward()
print("Grad of kernel parameters: ", list(kernel.parameters())[0].grad)

### test compound kernels
X = th.tensor([1.]).reshape(1,1).to(device)
Y = th.tensor([2.]).reshape(1,1).to(device)

l1 = np.ones(1,) * 0.7
rbf1 = RBF(1. ,l1, 1)

l2 = np.ones(1,) * 0.5
rbf2 = RBF(1., l2, 1)
cons_factor = 0.9
cons = Constant(0.9, 1)

assert th.isclose(rbf1(X, Y) * rbf2(X, Y), (rbf1 * rbf2)(X, Y))
assert th.isclose(rbf1(X, Y) + rbf2(X, Y), (rbf1 + rbf2)(X, Y))
assert th.isclose(rbf1(X, Y) ** 2., (rbf1 ** 2.)(X, Y))
assert th.isclose(rbf1(X, Y) ** 2. + rbf2(X, Y) ** 2., ((rbf1 ** 2.) + (rbf2 ** 2.))(X, Y))
assert th.isclose((rbf1(X, Y) ** 2. + rbf2(X, Y) ** 2.) * cons_factor, (((rbf1 ** 2.) + (rbf2 ** 2.)) * cons)(X, Y))
# show the operation logic table
print((((rbf1 ** 2.) + (rbf2 ** 2.)) * cons).get_parameters())

# seems all good ...