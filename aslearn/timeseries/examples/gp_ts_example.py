from asctr.system import Pendulum
from aslearn.common_utils.rollouts import collect_rollouts
from aslearn.nonparametric.gpr import ExactGPR
from aslearn.kernel.kernels import RBF
import torch as th
import numpy as np
import matplotlib.pyplot as plt
device = "cuda" if th.cuda.is_available() else "cpu"
pend = Pendulum()
traj_num = 6
traj_len = 50
X_l, U_l = collect_rollouts(system=pend, num=traj_num, traj_len=traj_len)

X_train = []
Y_train = []
for i in range(traj_num-1):
    X_temp = th.from_numpy(X_l[i][:-1]).to(device).double()
    X_temp = th.cat([X_temp, th.from_numpy(U_l[i]).to(device).double()], dim=1)
    X_train.append(X_temp)
    Y_train.append(th.from_numpy(X_l[i][1:]).to(device).double())
X_train = th.cat(X_train)
Y_train = th.cat(Y_train)

X_test = th.from_numpy(X_l[-1][0:1]).to(device).double().T
U_test = th.from_numpy(U_l[-1]).to(device).double()
Y_test = X_l[-1]

# GP SSM
l = np.ones([3, 2]) * 2.5
kernel = RBF(l=l, dim_in=3, dim_out=2)
gpr = ExactGPR(kernel=kernel)
gpr.fit(X_train, Y_train, call_hyper_opt=True)


forward_openloop = [X_test.detach().cpu().numpy().T]
forward_var = [np.zeros([1,2])]
for i in range(traj_len-1):
    x_test = th.cat([X_test, U_test[i:i+1]], dim=0)
    pred, var = gpr.predict(x_test.T, return_var=True)
    forward_openloop.append(pred.detach().cpu().numpy())
    forward_var.append((forward_var[-1] + (var.detach().cpu().numpy()))*(1+var.detach().cpu().numpy()) )
    X_test = pred.T
forward_openloop = np.concatenate(forward_openloop)
forward_var = np.concatenate(forward_var)

plt.plot(forward_openloop, 'b')
plt.plot(Y_test,'c')
plt.show()