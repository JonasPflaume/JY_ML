from aslearn.feature.global_features import PolynomialFT, FourierFT
import matplotlib.pyplot as plt
import numpy as np
from aslearn.timeseries.bsedmd import BSEDMD_1
import torch as th
device = "cuda" if th.cuda.is_available() else "cpu"

X = np.linspace(-15,15,300)[:,np.newaxis]
sw1 = np.sin(X)
sw2 = np.cos(X)
# sw3 = np.sin(0.4*X)
Y = np.concatenate([sw1, sw2], axis=1)
poly = PolynomialFT(degree=2)
four = FourierFT(degree=[1])
index = np.random.randint(low=0, high=300)
Xtrain, Ytrain = X[index:index+1], Y[index:index+1]
bounds = np.concatenate([-15*np.ones([1, 1]), 15*np.ones([1, 1])], axis=0)

for i in range(10):
    
    Xtrain_, Y_train_ = th.from_numpy(four(poly(Xtrain))).to(device).double(), th.from_numpy(Ytrain).to(device).double()

    predictor = BSEDMD_1().fit(Xtrain_, Y_train_, no_opt=False)
    # x_next = predictor.active_data_inquiry(predictor, [poly, four], box_cons=bounds, K=10)
    x_next = np.random.uniform(-15,15,size=(1,))
    x_next = x_next.reshape(1,-1)
    Xtrain = np.concatenate([Xtrain, x_next], axis=0)
    y_next = np.concatenate([np.sin(x_next), np.cos(x_next)], axis=1)
    # y_next = np.concatenate([np.sin(x_next)], axis=1)
    Ytrain = np.concatenate([Ytrain, y_next], axis=0)
    
    X_t = th.from_numpy(four(poly(X))).to(device).double()
    pred, var = predictor.predict(X_t, return_var=True)
    pred, var = pred.detach().cpu().numpy(), var.detach().cpu().numpy()
    
    del predictor
    print(np.linalg.norm(pred-Y))
    
plt.figure(figsize=[8,4])
plt.subplot(211)
plt.plot(X[:,0], Y[:,0], 'r.')
plt.plot(X[:,0], pred[:,0], 'b-')
plt.plot(Xtrain[:,0], np.zeros_like(Xtrain[:,0]), 'cx')
plt.plot(X[:,0], pred[:,0]+var[:,0,0], 'b-.')
plt.plot(X[:,0], pred[:,0]-var[:,0,0], 'b-.')

plt.subplot(212)
plt.plot(X[:,0], Y[:,1], 'r.')
plt.plot(X[:,0], pred[:,1], 'b-')
# plt.plot(Xtrain[:,0], np.zeros_like(Xtrain[:,0]), 'cx')
plt.plot(X[:,0], pred[:,1]+var[:,1,1], 'b-.')
plt.plot(X[:,0], pred[:,1]-var[:,1,1], 'b-.')


plt.legend()
plt.tight_layout()
plt.show()