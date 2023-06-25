from asctr.system import Pendulum
from aslearn.common_utils.rollouts import collect_rollouts
import torch as th
import numpy as np
from aslearn.parametric.ml_lr import ML_LR
from aslearn.feature.bellcurve import BellCurve
import matplotlib.pyplot as plt
device = "cuda" if th.cuda.is_available() else "cpu"

P = Pendulum()
traj_num = 10
traj_len = 100
X_l, U_l = collect_rollouts(P, num=traj_num, traj_len=traj_len)

X = np.zeros((traj_num,traj_len,2))
for i in range(traj_num):
    X[i,:,:] = X_l[i]
    
vali_num = 1
X_l, U_l = collect_rollouts(P, num=vali_num, traj_len=traj_len)
Xvali = np.zeros((vali_num,traj_len, 2))
for i in range(vali_num):
    Xvali[i,:,:] = X_l[i]
    
x_dim = X.shape[2]

bell = BellCurve(degree=200).fit(X.reshape(-1,x_dim))

X_later = X[:,1:,:].reshape(-1, x_dim)
X_input = X[:,:-1,:].reshape(-1, x_dim)

X_ = bell(X.reshape(-1,x_dim))
X_ori = X.reshape(-1,x_dim)
lift_dim = X_.shape[1]

X_later = bell(X_later).reshape(-1,lift_dim)
X_input = bell(X_input).reshape(-1,lift_dim)


X_ = th.from_numpy(X_).to(device).double()
X_ori = th.from_numpy(X_ori).to(device).double()
X_later = th.from_numpy(X_later).to(device).double()
X_input = th.from_numpy(X_input).to(device).double()


trans = ML_LR().fit(X_input, X_later)
# X_later_pred = trans.predict(X_input)
# X_later_ori = th.from_numpy(X[:,1:,:].reshape(-1, x_dim)).to(device).double()
back = ML_LR().fit(X_, X_ori)

x_test = Xvali[0]
x_test = bell(x_test)
x_test = th.from_numpy(x_test).to(device).double()

x0 = x_test[0:1,:]
traj = [Xvali[0][0:1]]
for i in range(100):
    x0 = trans.predict(x0)
    x_back = back.predict(x0).detach().cpu().numpy()
    traj.append(x_back)
traj = np.concatenate(traj)

plt.plot(traj,"r.")
plt.plot(Xvali[0],'b.')
plt.show()