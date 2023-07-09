import torch as th
from aslearn.base.regression import Regression
from typing import Optional
from aslearn.common_utils.check import RIGHT_SHAPE, PRINT, REPORT_VALUE
device = "cuda" if th.cuda.is_available() else "cpu"

class MLLR(Regression):
    ''' maximum likelihood multivariate linear regression
        estimation of weight and output noise 
    '''
    def __init__(self, nx:Optional[int]=None, ny:Optional[int]=None) -> None:
        super().__init__()
        if nx!=None and ny!=None:
            self.W = th.randn(ny, nx).double().to(device)
            self.v = th.randn(1, ny).double().to(device)
            self.beta_inv = th.randn(ny, ny).double().to(device)
            self.beta_inv = self.beta_inv.T @ self.beta_inv
            try:
                self.beta = th.linalg.cholesky(self.beta_inv)
            except:
                self.beta = th.linalg.cholesky(self.beta_inv + th.eye(self.beta_inv[0])*1e-6)
            self.beta = th.cholesky_inverse(self.beta)
                
    def fit(self,   X:th.Tensor, 
                    Y:th.Tensor, 
                    SX:Optional[th.Tensor]=None, 
                    SY:Optional[th.Tensor]=None, 
                    tolerance:Optional[float]=1e-8, 
                    max_iter:Optional[int]=20, 
                    info_level:Optional[int]=0):
        ''' use K-fold cross validation to get the penality
            X:      (N, feature)
            Y:      (N, output)
            SX:     (N, feature, ) # the covariance of input estimation
            SY:     (N, output, )  # only consider the diagonal case to save memory
        '''
        N = X.shape[0]
        nx = X.shape[1]
        ny = Y.shape[1]
        self.dim_feature = nx
        
        if SX != None:
            RIGHT_SHAPE(SX, (N, nx))
            SX = SX.sum(dim=0)
            XTX = X.T @ X + th.diag(SX)
        else:
            XTX = X.T @ X
        
        if SY != None:
            RIGHT_SHAPE(SY, (N, ny))
            SY = SY.sum(dim=0)
            YTY = Y.T @ Y + th.diag(SY)
        else:
            YTY = Y.T @ Y
            
        try:
            XTX_inv = th.linalg.cholesky(XTX)
        except:
            XTX_inv = th.linalg.cholesky(XTX + 1e-6*th.eye(XTX.shape[0]).double().to(device))
            
        XTX_inv = th.cholesky_inverse(XTX_inv)
        # initialization
        N = X.shape[0]
        self.W = 1e-6*th.randn(ny, nx).double().to(device)
        self.v = 1e-6*th.randn(1, ny).double().to(device)
        self.beta_inv = 1e-6*th.randn(ny, ny).double().to(device)
        
        # start training
        stop_flag = False
        counter = 1
        while not stop_flag:
            new_v = 1/N * (Y - X @ self.W.T).sum(dim=0, keepdim=True)
            res_tolerance = th.max(th.abs(new_v - self.v))
            self.v = new_v
            VI = self.v.repeat(N, 1)
            new_W = (Y.T @ X - VI.T @ X) @ XTX_inv
            res_tolerance = th.max(th.tensor([th.max(th.abs(new_W - self.W)), res_tolerance]))
            self.W = new_W
            # check stop criterion
            beta_inv_new = 1/N * (YTY - Y.T@X@self.W.T - Y.T@VI - self.W@X.T@Y + self.W@XTX@self.W.T + self.W@X.T@VI - VI.T@Y + VI.T@X@self.W.T + VI.T@VI)
            res_tolerance = th.max(th.tensor([th.max(th.abs(beta_inv_new - self.beta_inv)), res_tolerance]))
            self.beta_inv = beta_inv_new
            PRINT("Interation: {0} ".format(counter), info_level)
            REPORT_VALUE(res_tolerance.item(), "Curr_tol: ", info_level)

            if res_tolerance < tolerance or counter >= max_iter:
                stop_flag = True
            
            counter += 1
        try:
            self.beta = th.linalg.cholesky(self.beta_inv)
        except:
            self.beta = th.linalg.cholesky(self.beta_inv + 1e-6*th.eye(self.beta_inv.shape[0]).double().to(device))
        self.beta = th.cholesky_inverse(self.beta)
        return self
    
    def predict(self, x:th.Tensor, return_std:Optional[bool]=False):
        '''
        '''
        RIGHT_SHAPE(x, (-1, self.dim_feature))
        with th.no_grad():
            mean = x @ self.W.T + self.v
            if return_std:
                var = th.diag(self.beta_inv).reshape(1,-1)
                return mean, th.sqrt(var)
            return mean
        
    
if __name__ == "__main__":
    import numpy as np
    X = th.linspace(-5,5,100).reshape(-1,1).double().to(device)
    Y1 = 1.5*X + 1. + th.randn_like(X) * 1.2
    Y2 = 0.3*X + 2. + th.randn_like(X) * 0.5
    Y = th.cat([Y1, Y2], dim=1)
    blr = MLLR().fit(X, Y, info_level=1)
    print("Ground truth W: ", 1.5, 0.3, " Est W: %.4f, %.4f" % (blr.W[0].item(), blr.W[1].item()))
    print("Ground truth sigma: ", 1.2, 0.5, " Est sigma: %.4f, %.4f" % (th.sqrt(blr.beta_inv[0,0]).item(), 
                                                                        th.sqrt(blr.beta_inv[1,1]).item()))
    print("Ground truth offset: ", 1., 2., " Est offset: %.4f, %.4f" % (blr.v[0,0].item(), 
                                                                        blr.v[0,1].item()))
    
    import matplotlib.pyplot as plt
    
    Xt = th.linspace(-7,7,200).reshape(-1,1).double().to(device)
    pred, std = blr.predict(Xt, return_std=True)
    pred, std = pred.detach().cpu().numpy(), std.detach().cpu().numpy()
    Xt = Xt.detach().cpu().numpy()
    plt.plot()
    plt.plot(Xt, pred, '-b')
    plt.plot(Xt, pred+1.96*std, '-.b')
    plt.plot(Xt, pred-1.96*std, '-.b')
    plt.plot(X.detach().cpu().numpy(), Y1.detach().cpu().numpy(), '.r')
    plt.plot(X.detach().cpu().numpy(), Y2.detach().cpu().numpy(), '.r')
    plt.show()
    
    
    