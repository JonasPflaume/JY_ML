import torch
from regression import Regression
import numpy as np
import matplotlib.pyplot as plt

class RidgeReg(Regression):
    ''' ridge regression, hyperparameter determined by cross validation
    '''
    def __init__(self) -> None:
        super().__init__()
        self.labd = torch.logspace(-7,7,14)
        
    def fit(self, X, Y, K=6, plot=False):
        ''' use K-fold cross validation to get the penality
            X: (N, feature)
            Y: (N, output)
            K: K-fold CV
        '''
        K_index = torch.randperm(len(X))
        K_len = len(X) // K
        loss_mean, loss_var = [], []
        for labd in self.labd:
            loss_K = []
            for k in range(K):
                val_index = K_index[k*K_len:(k+1)*K_len]
                if k == K-1:
                    val_index = K_index[k*K_len:]
                train_index = torch.from_numpy(np.setdiff1d(K_index.numpy(), val_index.numpy()))
                X_train, Y_train = X[train_index], Y[train_index]
                X_val, Y_val = X[val_index], Y[val_index]
                I_diag = torch.eye(X_train.shape[1]) * labd
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
        I_diag = torch.eye(X_train.shape[1]) * labd_best
        I_diag[0,0] = 0.
        self.weight = torch.inverse(X_train.T @ X_train + I_diag) @ X_train.T @ Y_train
        
        if plot:
            plt.plot(self.labd, loss_mean, '-r')
            plt.plot(self.labd, loss_mean+loss_var, '-.r')
            plt.plot(self.labd, loss_mean-loss_var, '-.r')
            plt.plot(self.labd[torch.argmin(loss_mean)], 0, 'xc')
            plt.xscale("log")
            plt.xlabel("labd")
            plt.ylabel("validation loss")
            plt.show()
        return self
    
    def predict(self, x):
        return x @ self.weight
    
if __name__ == "__main__":
    from jylearn.data.reg_data import polynomial_data
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import PolynomialFeatures
    #### data
    X, Y = polynomial_data()
    for order in range(1, 20):
        
        poly = PolynomialFeatures(order)
        X_poly = poly.fit_transform(X)
        #### training and testing
        X_poly, Y_ = torch.from_numpy(X_poly), torch.from_numpy(Y)
        
        X_test = np.linspace(np.min(X), np.max(X), 50).reshape(-1,1)
        X_test_poly = poly.fit_transform(X_test)

        X_test_poly = torch.from_numpy(X_test_poly)
        rr = RidgeReg().fit(X_poly, Y_, plot=True)
        pred = rr.predict(X_test_poly)
        
        beta = torch.inverse(X_poly.T @ X_poly) @ X_poly.T @ Y
        pred_ls = X_test_poly @ beta
        plt.plot(X.squeeze(), Y.squeeze(), '.r')
        plt.plot(X_test.squeeze(), pred.squeeze(), '-b')
        plt.plot(X_test.squeeze(), pred_ls.squeeze(), '-c')
        plt.show()