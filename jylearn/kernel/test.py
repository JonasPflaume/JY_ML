from pickletools import read_bytes1
import torch as th
import torch.nn as nn
from kernels import RBF, Constant, DotProduct, White
import numpy as np
th.manual_seed(0)
device = "cuda" if th.cuda.is_available() else "cpu"


#### test grad computation
print("....................Test grad .....................")
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
print("....................Test compound .....................")
X = th.tensor([1.]).reshape(1,1).to(device).double()
Y = th.tensor([2.]).reshape(1,1).to(device).double()

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

### test autograd of compound kernels
print("....................Test grad .....................")
kernel = (((rbf1 ** 2.) + (rbf2 ** 2.)) * cons)
res = kernel(X, Y)
res.backward()
for name, param in kernel.named_parameters():
    print("The gradient of "+name, param.grad)

# test the generalized dot product kernel
print("....................Test innerproduct .....................")
X = th.tensor([1., 1.]).reshape(1,2).to(device).double()
Y = th.tensor([2., 2.]).reshape(1,2).to(device).double()
dot = DotProduct(dim=2, l=np.array([0.5, 0.1]))
assert th.isclose(dot(X, Y), th.tensor([[1.2]]).to(device).double())

# test white kernel
print("....................Test white .....................")
X = th.tensor([1., 1., 1., 1.]).reshape(2,2).to(device).double()
Y = th.tensor([2., 2., 2., 2.]).reshape(2,2).to(device).double()
white = White(dim=2, c=0.5)
assert th.all(white(X, X) == th.eye(2).to(device) * 0.5)
assert th.all(white(X, Y) == th.zeros(2,2).to(device))

# seems all good ...