from jylearn.parametric.regression import Regression
import torch as th
device = "cuda" if th.cuda.is_available() else "cpu"
th.pi = th.acos(th.zeros(1)).item() * 2

class ExactGPR(Regression):
    
    def __init__(self, kernel):
        '''
            kernel:
                    the kernel function
        '''
        self.kernel = kernel

    def fit(self, X, Y, call_hyper_opt=False):
        '''
        '''
        if len(X.shape) == 1:
            # in case the data is 1-d
            X = X.reshape(-1,1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1,1)
        n = X.shape[0]
        d = X.shape[1]
        m = Y.shape[1] # output dimension
        self._X = X
        # call the hyperparameter optimization
        if call_hyper_opt:
            pass
        
        K = self.kernel(X, X) # K has shape (n,n) number of data
        try:
            u = th.cholesky(K)
        except:
            print("The cho_factor meet singular matrix, now add damping...")
            K += th.eye(K.shape[0]).to(device) * 1e-8
            u = th.cholesky(K)
        
        self.alpha = th.cholesky_solve(Y, u) # (n,n_y) yd-output space dim, important -> cho_solve
        self.L = u
        
        # print the margianl likelyhood
        evidence = - 0.5 * th.einsum("ik,ik->k", Y, self.alpha) - th.log(th.diagonal(self.L)).sum() - n / 2 * th.log(2*th.tensor([th.pi]).to(device))
        evidence = evidence.sum(axis=-1)
        print("The evidence is: ", evidence)
        
        del K, u, evidence
        th.cuda.empty_cache()
    
    def predict(self, x, return_var=False):
        '''
        '''
        if len(x.shape) == 1:
            # in case the data is 1-d
            x = x.reshape(-1,1)
            
        k = self.kernel(x, self._X) # (n_x, n)
        mean = k @ self.alpha
        if return_var:
            v = th.triangular_solve(k.T, self.L, upper=False)[0]
            prior_std = self.kernel.diag(x)
            var = prior_std - th.einsum("ij,ji->i", v.T, v)
            
            del k, v, prior_std
            th.cuda.empty_cache()
            return mean, var.reshape(-1,1)
        else:
            del k
            th.cuda.empty_cache()
            return mean
        
if __name__ == "__main__":
    # test
    import matplotlib.pyplot as plt
    from jylearn.kernel.kernels import RBF, White, Matern
    import numpy as np
    th.manual_seed(0)
    
    l = np.ones(1,) * 0.7
    kernel = RBF(dim=1, l=l, sigma=1.)
    gpr = ExactGPR(kernel=kernel)
    
    train_data_num = 20
    X = th.linspace(-5,5,100).reshape(-1,1).to(device).double()
    Y = th.cos(X)
    Xtrain = th.linspace(-5,5,train_data_num).reshape(-1,1).to(device).double()
    Ytrain = th.cos(Xtrain) + th.randn(train_data_num,1).to(device).double() * 0.2
    
    gpr.fit(Xtrain, Ytrain)
    
    import time
    s = time.time()
    for i in range(100):
        mean, var = gpr.predict(X, return_var=True)
    e = time.time()
    print("The time for each pred: %.5f" % ((e-s)/100))
    plt.plot(X.detach().cpu().numpy(), mean.detach().cpu().numpy(), label="Prediction")
    plt.plot(X.detach().cpu().numpy(), mean.detach().cpu().numpy() + var.detach().cpu().numpy(), '-.r', label="Var")
    plt.plot(X.detach().cpu().numpy(), mean.detach().cpu().numpy() - var.detach().cpu().numpy(), '-.r')
    plt.plot(X.detach().cpu().numpy(), Y.detach().cpu().numpy(), label="GroundTueth")
    plt.plot(Xtrain.detach().cpu().numpy(), Ytrain.detach().cpu().numpy(), 'rx', label="data", alpha=0.3)
    plt.grid()
    plt.legend()
    plt.show()