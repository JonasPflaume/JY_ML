import matplotlib.pyplot as plt
import numpy as np
import torch as th
from aslearn.parametric.vrvm import VRVM
from aslearn.parametric.ridge import RidgeReg

from aslearn.feature.global_features import PolynomialFT
from aslearn.feature.global_features import FourierFT
from aslearn.feature.global_features import SquareWaveFT
from aslearn.feature.bellcurve import BellCurve

device = "cuda" if th.cuda.is_available() else "cpu"
th.cuda.empty_cache()

# let's fit this highly nonlinear square wave function
X = np.linspace(-5,5,300)[:,np.newaxis]
sw1 = np.sin(2*np.pi*0.8*X) - 2*np.cos(2*np.pi*0.4*X)
sw2 = np.cos(2*np.pi*0.8*X) + 2*np.sin(2*np.pi*0.4*X)
sw3 = np.sin(2*np.pi*0.4*X)
Y = np.concatenate([sw1, sw2, sw3], axis=1) + np.random.randn(300,3) * 0.1

poly = PolynomialFT(degree=2)
# sqw = SquareWaveFT(frequencies=np.linspace(0.1,10,100))
fri = FourierFT(degree=np.linspace(0.1,15,15))
bell = BellCurve(degree=10, l=0.7).fit(X)

X_f = poly(fri(X)) 
print("Feauture dim: ", X_f.shape[1])
X_t, Y_t = th.from_numpy(X_f).to(device).double(), th.from_numpy(Y).to(device).double()

blr = VRVM(X_t.shape[1], 3).fit(X_t[100:200], Y_t[100:200])
# use a fraction of data
# only 1/3 data, we acieved a superb generalization that ridge regression
# because the sparse model have pushed the weights of 
# nonrelevant features to 0.
                                                  
rr = RidgeReg().fit(X_t[100:200], Y_t[100:200])

# blr.active_query()

pred, var = blr.predict(X_t, return_var=True)
pred, var = pred.detach().cpu().numpy(), var.detach().cpu().numpy()

pred_ridge = rr.predict(X_t)
pred_ridge = pred_ridge.detach().cpu().numpy()

print("BLR: {:.2f}".format(np.linalg.norm(pred-Y, axis=1).sum()))
print("ridge: {:.2f}".format(np.linalg.norm(pred_ridge-Y, axis=1).sum()))

plt.figure(figsize=[5,7])
plt.subplot(311)
plt.plot(X[:,0], Y[:,0], 'r.')
plt.plot(X[:,0], pred[:,0], 'b-')
# plt.plot(X[:,0], pred_ridge[:,0], 'c-')
plt.plot(X[:,0], pred[:,0]+3*var[:,0], 'b-.')
plt.plot(X[:,0], pred[:,0]-3*var[:,0], 'b-.')
plt.subplot(312)
plt.plot(X[:,0], Y[:,1], 'r.')
plt.plot(X[:,0], pred[:,1], 'b-')
# plt.plot(X[:,0], pred_ridge[:,1], 'c-')
plt.plot(X[:,0], pred[:,1]+3*var[:,1], 'b-.')
plt.plot(X[:,0], pred[:,1]-3*var[:,1], 'b-.')
plt.subplot(313)
plt.plot(X[:,0], Y[:,2], 'r.')
plt.plot(X[:,0], pred[:,2], 'b-', label="vRVM")
# plt.plot(X[:,0], pred_ridge[:,2], 'c-', label="Ridge")
plt.plot(X[:,0], pred[:,2]+3*var[:,2], 'b-.')
plt.plot(X[:,0], pred[:,2]-3*var[:,2], 'b-.')
plt.legend()
plt.tight_layout()
plt.show()