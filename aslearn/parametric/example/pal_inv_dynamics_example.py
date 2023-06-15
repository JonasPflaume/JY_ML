from asctr.models.panda.C_matrix import C_panda
from asctr.models.panda.G_vector import G_panda
from asctr.models.panda.M_matrix import M_panda
from asctr.models.panda.param import get_param_CMA
import numpy as np
from numpy import cos, sin

from aslearn.feature.global_features import PolynomialFT
import matplotlib.pyplot as plt
import numpy as np
from aslearn.parametric.vrvm import VRVM
import torch as th
device = "cuda" if th.cuda.is_available() else "cpu"

param = get_param_CMA()

def get_inv_torque(x):
    if len(x.shape) == 2:
        q = x[0,:7]
        dq = x[0,7:14]
        ddq = x[0,14:21]
    else:
        q = x[:7]
        dq = x[7:14]
        ddq = x[14:21]

    M = M_panda(param, q, cos, sin)
    C = C_panda(param, q, dq, cos, sin)
    G = G_panda(param, q, cos, sin)

    M, C, G = np.array(M).reshape(7,7), np.array(C).reshape(7,7), np.array(G).reshape(7,1)
    return M @ ddq.reshape(-1,1) + C @ dq.reshape(-1,1) + G


max_q = np.array([2.8973,1.7628,2.8973,-0.0698,2.8973,3.7525,2.8973]).reshape(1,-1)
min_q = -max_q.copy()
min_q[0,3] = -3.0718
min_q[0,5] = -0.0175

max_dq = np.array([2.175]*4+[2.61]*3).reshape(1,-1)
min_dq = -max_dq.copy()

max_ddq = np.array([15,7.5,10,12.5,15,20,20]).reshape(1,-1)
min_ddq = -max_ddq.copy()

min_x = np.concatenate([min_q, min_dq, min_ddq], axis=1)
max_x = np.concatenate([max_q, max_dq, max_ddq], axis=1)
bounds = np.concatenate([min_x, max_x], axis=0)


Xvali, Yvali = [], []
for i in range(1000):
    Xtemp = np.random.uniform(low=bounds[0], high=bounds[1]).reshape(1,-1)
    Ytemp = get_inv_torque(Xtemp).reshape(1,-1)
    Xvali.append(Xtemp)
    Yvali.append(Ytemp)
    
Xvali, Yvali = np.concatenate(Xvali), np.concatenate(Yvali)


from aslearn.nonparametric.gpr import ExactGPR
from aslearn.kernel.kernels import RBF, White

x_next = np.random.uniform(low=bounds[0], high=bounds[1])
x_next = x_next.reshape(1,-1)
y_next = get_inv_torque(x_next).reshape(1,-1)
Xtrain, Ytrain = [x_next], [y_next]
Xtrain, Ytrain = np.concatenate(Xtrain), np.concatenate(Ytrain)
Xtrain, Ytrain = th.from_numpy(Xtrain).double().to(device), th.from_numpy(Ytrain).double().to(device)
Xvali, Yvali = th.from_numpy(Xvali).double().to(device), th.from_numpy(Yvali).double().to(device)

kernel = RBF(25*np.ones([21, 7]), 21, 7) + White(0.1*np.ones([7,]), 21, 7)
gpr = ExactGPR(kernel=kernel).fit(Xtrain, Ytrain)

pred = gpr.predict(Xvali, return_var=False)
pred = pred.detach().cpu().numpy()
ori_error = np.abs(pred-Yvali.detach().cpu().numpy()).sum(axis=0)/1000

for i in range(1000):
    
    # if i % 20 == 0:
    # pred = gpr.predict(Xvali, return_var=False)
    # pred = pred.detach().cpu().numpy()
    # print("Round", i , "Improvement: ", ori_error - np.abs(pred-Yvali.detach().cpu().numpy()).sum(axis=0)/1000)
    # print(gpr.kernel)
    
    # x_next = gpr.maximum_entropy_point(box_cons=bounds, K=5).reshape(1,-1)
    x_next = np.random.uniform(low=bounds[0], high=bounds[1]).reshape(1,-1)
    y_next = get_inv_torque(x_next).reshape(1,-1)
    x_next, y_next = th.from_numpy(x_next).double().to(device), th.from_numpy(y_next).double().to(device)
    
    Xtrain = th.cat([Xtrain, x_next], dim=0)
    Ytrain = th.cat([Ytrain, y_next], dim=0)
    
    # kernel = RBF(20*np.ones([21, 7]), 21, 7) + White(np.ones([7,]), 21, 7)
    # gpr = ExactGPR(kernel=kernel).fit(Xtrain, Ytrain, verbose=True)
    # if i % 20 == 0:
    #     gpr.fit(Xtrain, Ytrain, call_hyper_opt=True, verbose=True)
    # else:
    #     gpr.fit(Xtrain, Ytrain, call_hyper_opt=False, verbose=True)

kernel = RBF(100*np.abs(np.random.randn(21,7)),21,7) + White(np.ones([7,]),21,7)
gpr = ExactGPR(kernel=kernel).fit(Xtrain, Ytrain, verbose=True)

pred = gpr.predict(Xvali, return_var=False)
pred = pred.detach().cpu().numpy()
print(gpr.kernel)
print("Round", i , "Improvement: ", ori_error - np.abs(pred-Yvali.detach().cpu().numpy()).sum(axis=0)/1000)
