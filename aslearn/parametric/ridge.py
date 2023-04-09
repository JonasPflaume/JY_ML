import torch
from aslearn.parametric.regression import Regression
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"

class RidgeReg(Regression):
    ''' ridge regression, hyperparameter determined by cross validation
    '''
    def __init__(self) -> None:
        super().__init__()
        self.labd = torch.logspace(-4,6,10).to(device)
        
    def fit(self, X, Y, K=5, plot_vali_loss=False):
        ''' use K-fold cross validation to get the penality
            X:      (N, feature)
            Y:      (N, output)
            K:      K-fold CV
        '''
        K_index = torch.randperm(len(X))
        K_len = len(X) // K
        loss_mean, loss_var = [], []
        for labd in tqdm( self.labd ):
            loss_K = []
            for k in range(K):
                val_index = K_index[k*K_len:(k+1)*K_len]
                if k == K-1:
                    val_index = K_index[k*K_len:]
                train_index = torch.from_numpy(np.setdiff1d(K_index.numpy(), val_index.numpy()))
                X_train, Y_train = X[train_index].to(device), Y[train_index].to(device)
                X_val, Y_val = X[val_index].to(device), Y[val_index].to(device)
                I_diag = torch.eye(X_train.shape[1]).to(device) * labd
                I_diag[0,0] = 0.
                weight = torch.inverse(X_train.T @ X_train + I_diag) @ X_train.T @ Y_train
                
                pred = X_val @ weight
                loss = torch.norm(pred - Y_val, dim=1).sum() / len(Y_val)
                loss_K.append(loss)
            loss_K = torch.tensor(loss_K)
            loss_mean.append(loss_K.mean())
            loss_var.append(loss_K.var())
        loss_mean, loss_var = torch.tensor(loss_mean), torch.tensor(loss_var)
        labd_best = self.labd[torch.argmin(loss_mean)]
        I_diag = torch.eye(X.shape[1]).to(device) * labd_best
        I_diag[0,0] = 0.
        self.weight = torch.inverse(X.T @ X + I_diag) @ X.T @ Y
        
        if plot_vali_loss:
            labd_l, loss_mean, loss_var = self.labd.to("cpu").numpy(), loss_mean.to("cpu").numpy(), loss_var.to("cpu").numpy()
            plt.plot(labd_l, loss_mean, '-r')
            plt.plot(labd_l, loss_mean+loss_var, '-.r')
            plt.plot(labd_l, loss_mean-loss_var, '-.r')
            plt.plot(labd_l[np.argmin(loss_mean)], 0, 'xc')
            plt.xscale("log")
            plt.xlabel("labd")
            plt.ylabel("validation loss")
            plt.show()
        return self
    
    def predict(self, x):
        return x @ self.weight