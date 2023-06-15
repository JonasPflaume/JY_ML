from aslearn.kernel.kernels import RBF, White
import matplotlib.pyplot as plt
import numpy as np
from aslearn.nonparametric.gpr import ExactGPR
import torch as th
device = "cuda" if th.cuda.is_available() else "cpu"

X = np.linspace(-15,15,300)[:,np.newaxis]
sw1 = np.sin(X)
sw2 = np.cos(X)
# sw3 = np.sin(0.4*X)
Y = np.concatenate([sw1, sw2], axis=1)

bounds = np.concatenate([-15*np.ones([1, 1]), 15*np.ones([1, 1])], axis=0)
l = np.ones([1, 2]) * 1.5
c = np.array([0.1, 0.1])

loss_random = []
loss_al = []
for j in range(2):
    kernel = White(c=c, dim_in=1, dim_out=2) + RBF(l=l, dim_in=1, dim_out=2)
    gpr = ExactGPR(kernel=kernel)
    index = np.random.randint(low=0, high=300)
    Xtrain, Ytrain = X[index:index+1], Y[index:index+1]
    
    for i in range(50):
        Xtrain_, Y_train_ = th.from_numpy(Xtrain).to(device).double(), th.from_numpy(Ytrain).to(device).double()
        gpr.fit(Xtrain_, Y_train_)
        if j == 0:
            x_next = gpr.maximum_entropy_point(box_cons=bounds,K=15)
        elif j==1:
            x_next = np.random.uniform(-15,15,size=(1,))
        x_next = x_next.reshape(1,-1)
        Xtrain = np.concatenate([Xtrain, x_next], axis=0)
        y_next = np.concatenate([np.sin(x_next), np.cos(x_next)], axis=1)
        # y_next = np.concatenate([np.sin(x_next)], axis=1)
        Ytrain = np.concatenate([Ytrain, y_next], axis=0)
        
        X_t = th.from_numpy(X).to(device).double()
        pred, var = gpr.predict(X_t, return_var=True)
        pred, var = pred.detach().cpu().numpy(), var.detach().cpu().numpy()
        
        if j == 0:
            loss_al.append(np.linalg.norm(pred-Y))
        elif j==1:
            loss_random.append(np.linalg.norm(pred-Y))
    
plt.figure(figsize=[8,4])
# plt.subplot(311)
# plt.plot(X[:,0], Y[:,0], 'r.')
# plt.plot(X[:,0], pred[:,0], 'b-')
# plt.plot(Xtrain[:,0], np.zeros_like(Xtrain[:,0]), 'cx')
# plt.plot(X[:,0], pred[:,0]+var[:,0], 'b-.')
# plt.plot(X[:,0], pred[:,0]-var[:,0], 'b-.')

# plt.subplot(312)
# plt.plot(X[:,0], Y[:,1], 'r.')
# plt.plot(X[:,0], pred[:,1], 'b-')
# # plt.plot(Xtrain[:,0], np.zeros_like(Xtrain[:,0]), 'cx')
# plt.plot(X[:,0], pred[:,1]+var[:,1], 'b-.')
# plt.plot(X[:,0], pred[:,1]-var[:,1], 'b-.')

plt.plot(loss_al, 'r', label="AL")
plt.plot(loss_random, 'b', label="Random")
plt.grid()
plt.xlabel("Sampling number")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.show()