import torch as th
import torch.nn as nn
from kernels import Parameters
from kernels import KernelOperation

### Test the parameter group
t = th.zeros(3,)
P1 = Parameters("1a", nn.parameter.Parameter(t))

P2 = Parameters("2a", nn.parameter.Parameter(t))

P1.join(P2, KernelOperation.ADD)

t = th.zeros(3,)
P2 = Parameters("3a", nn.parameter.Parameter(t))
P2.join(P1, KernelOperation.ADD)

t = th.zeros(1,)
P3 = Parameters("4a", nn.parameter.Parameter(t), requres_grad=False)
P2.join(P3, KernelOperation.EXP)
print(P2)

### Seems OK