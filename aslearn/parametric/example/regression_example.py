import matplotlib.pyplot as plt
import numpy as np
import torch as th
from aslearn.parametric.vrvm import VRVM
from aslearn.parametric.ridge import RidgeReg
from aslearn.nonparametric.gpr import ExactGPR

from aslearn.kernel.kernels import RBF, White
from aslearn.feature.global_features import PolynomialFT, FourierFT, SquareWaveFT
from aslearn.feature.bellcurve import BellCurve

device = "cuda" if th.cuda.is_available() else "cpu"
th.cuda.empty_cache()

## prepare the data
X = np.linspace(-5,5,300)[:,np.newaxis]
sw1 = 0.5*np.sin(1.5*np.pi*0.8*X) - 1.2*np.cos(1.5*np.pi*0.4*X)
sw2 = 0.5*np.cos(1.5*np.pi*0.8*X) + 1.2*np.sin(1.5*np.pi*0.4*X)
sw3 = np.sin(2*np.pi*0.4*X)
Y = np.concatenate([sw1, sw2, sw3], axis=1) + np.random.randn(300,3) * 0.35

poly = PolynomialFT(degree=1)
# sqw = SquareWaveFT(frequencies=np.linspace(0.1,10,100))
fri = FourierFT(degree=np.linspace(0.1,7,20))
bell = BellCurve(degree=10, l=0.7).fit(X)

X_f = poly(fri(X))
print("Feauture dim: ", X_f.shape[1])
X_t, Y_t = th.from_numpy(X_f).to(device).double(), th.from_numpy(Y).to(device).double()

## fit the model
blr = VRVM(X_t.shape[1], 3).fit(X_t[100:201], Y_t[100:201])
rr = RidgeReg().fit(X_t[100:201], Y_t[100:201])

kernel = RBF(np.ones([1, 3])*2.1, 1, 3) + White(np.ones([3,])*10., 1, 3)

X_gpr, Y_gpr = th.from_numpy(X).to(device).double(), th.from_numpy(Y).to(device).double()
gpr = ExactGPR(kernel=kernel)
gpr.fit(X_gpr[100:201], Y_gpr[100:201], call_hyper_opt=True, solver="LBFGS")
print(kernel)
# use a fraction of data
# only 1/3 data, we acieved a superb generalization that ridge regression
# because the sparse model have pushed the weights of 
# nonrelevant features to 0.

## make predictions
pred, var = blr.predict(X_t, return_var=True)
pred, var = pred.detach().cpu().numpy(), var.detach().cpu().numpy()

pred_ridge = rr.predict(X_t)
pred_ridge = pred_ridge.detach().cpu().numpy()

pred_gpr, var_gpr = gpr.predict(X_gpr, return_var=True)
pred_gpr, var_gpr = pred_gpr.detach().cpu().numpy(), var_gpr.detach().cpu().numpy()


print("BLR: {:.2f}".format(np.linalg.norm(pred-Y, axis=1).sum()))
print("ridge: {:.2f}".format(np.linalg.norm(pred_ridge-Y, axis=1).sum()))
print("GPR: {:.2f}".format(np.linalg.norm(pred_gpr-Y, axis=1).sum()))

plt.figure(figsize=[13,4])
plt.subplot(131)
plt.plot(X[:,0], Y[:,0], 'r.')
plt.plot(X[:,0], pred[:,0], 'b-')
plt.plot(X[:,0], pred_ridge[:,0], 'c-')
plt.fill_between(X[:,0], pred[:,0]-var[:,0], pred[:,0]+var[:,0], color='b', alpha=0.4)
# plt.plot(X[:,0], pred[:,0]-var[:,0], 'b-.')

plt.plot(X[:,0], pred_gpr[:,0], 'y-')
plt.fill_between(X[:,0], pred_gpr[:,0]-var_gpr[:,0], pred_gpr[:,0]+var_gpr[:,0], color='y', alpha=0.4)
# plt.plot(X[:,0], pred_gpr[:,0]-var_gpr[:,0], 'y-.')

plt.subplot(132)
plt.plot(X[:,0], Y[:,1], 'r.')
plt.plot(X[:,0], pred[:,1], 'b-')
plt.plot(X[:,0], pred_ridge[:,1], 'c-')
plt.fill_between(X[:,0], pred[:,1]-var[:,1], pred[:,1]+var[:,1], color='b', alpha=0.4)
# plt.plot(X[:,0], pred[:,1]+var[:,1], 'b-.')
# plt.plot(X[:,0], pred[:,1]-var[:,1], 'b-.')

plt.plot(X[:,0], pred_gpr[:,1], 'y-')
# plt.plot(X[:,0], pred_gpr[:,1]+var_gpr[:,1], 'y-.')
# plt.plot(X[:,0], pred_gpr[:,1]-var_gpr[:,1], 'y-.')
plt.fill_between(X[:,0], pred_gpr[:,1]-var_gpr[:,1], pred_gpr[:,1]+var_gpr[:,1], color='y', alpha=0.4)

plt.subplot(133)
plt.plot(X[:,0], Y[:,2], 'r.', label="data")
plt.plot(X[:,0], pred[:,2], 'b-', label="vRVM")
plt.plot(X[:,0], pred_ridge[:,2], 'c-', label="Ridge")
# plt.plot(X[:,0], pred[:,2]+var[:,2], 'b-.')
# plt.plot(X[:,0], pred[:,2]-var[:,2], 'b-.')
plt.fill_between(X[:,0], pred[:,2]-var[:,2], pred[:,2]+var[:,2], color='b', alpha=0.4)

plt.plot(X[:,0], pred_gpr[:,2], 'y-', label="gpr")
# plt.plot(X[:,0], pred_gpr[:,2]+var_gpr[:,2], 'y-.')
# plt.plot(X[:,0], pred_gpr[:,2]-var_gpr[:,2], 'y-.')
plt.fill_between(X[:,0], pred_gpr[:,2]-var_gpr[:,2], pred_gpr[:,2]+var_gpr[:,2], color='y', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.savefig('res.svg')
plt.show()