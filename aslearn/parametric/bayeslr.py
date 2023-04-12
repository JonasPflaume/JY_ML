import torch as th
from aslearn.base.regression import Regression
device = "cuda" if th.cuda.is_available() else "cpu"

class BayesLR(Regression):
    ''' Fully Bayesian linear regression,
        the hyperparameters are optimized through mean-field variational inference.
        the prior of weight and output noise are assumed to be fully decoupled Gaussian.
        We treat the prior of weight as diagonal Gaussian and the prior of precision as a multiplication of Gamma distributions,
        this leads to an equvalent representation to the variational Relevance Vector Machine (V-RVM) for multivariate regression.
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, Y, iter_num=100):
        '''
        '''
        # number of data, feature dimension
        N, nx = X.shape
        # output dimension 
        ny = Y.shape[1]
        # initialize model parameter
        # q(alpha): prior of weight precision, product of gamma distributions, 
        # gamma_a:(ny,nx), gamma_b:(ny,nx)
        # initialized to near 0. is usually a good choice
        gamma_aN = 10e-8 * th.ones(ny, nx).to(device).double()
        gamma_bN = 10e-8 * th.ones(ny, nx).to(device).double()
        gamma_a0 = gamma_aN.clone()
        gamma_b0 = gamma_bN.clone()
        
        # q(beta): prior of precision of output noise
        # gamma_c:(ny,1), gamma_d:(ny,1)
        # initialized to near 0. is usually a good choice
        gamma_cN = 10e-8 * th.ones(ny, 1).to(device).double()
        gamma_dN = 10e-8 * th.ones(ny, 1).to(device).double()
        gamma_c0 = gamma_cN.clone()
        gamma_d0 = gamma_dN.clone()
        
        # q(omega): prior of weights
        meanN = th.zeros(ny, nx).to(device).double()
        covarianceN = th.zeros(ny, nx, nx).to(device).double()
        
        # coordinate decent estimation
        for _ in range(iter_num):
            # expect terms
            E_beta = gamma_cN / gamma_dN # (ny,1)
            E_diag_alpha = gamma_aN / gamma_bN # don't exand this term into diag matrices (ny,nx)
            
            # update weights q(omega)
            cov_temp = E_beta.unsqueeze(dim=2) * X.T @ X # (ny,nx,nx), utilize the broadcast of torch
            cov_temp[:,th.arange(nx),th.arange(nx)] += E_diag_alpha # add E_diag_alpha to diagonal
            try:
                L_cov_temp = th.linalg.cholesky(cov_temp)
                covarianceN = th.cholesky_inverse(L_cov_temp) # (ny,nx,nx)
            except:
                raise ValueError("Try to change initial value of each priors.")
            
            mean_temp = X.T @ Y # (nx, ny)
            mean_temp = th.einsum("bij,jb->bi", covarianceN, mean_temp) # (ny,nx)
            meanN = E_beta * mean_temp
            
            # update q(alpha)
            gamma_aN = gamma_a0 + 0.5
            gamma_bN = gamma_b0 + 0.5 * covarianceN[:,th.arange(nx),th.arange(nx)] + 0.5 * meanN ** 2.
            
            # update q(beta)
            gamma_cN = gamma_c0 + N / 2.0
            
            dN_temp1 = Y.T - th.einsum("ij,bj->bi", X, meanN) # (ny,N)
            dN_temp1 = th.linalg.norm(dN_temp1, dim=1)**2. # (ny,)
            
            dN_temp2 = th.einsum("ij,bjk->bik", X.T @ X, covarianceN) # (ny,nx,nx)
            dN_temp2 = th.einsum("bii->b", dN_temp2) # (ny,) calc trace

            gamma_dN = gamma_d0 + 0.5 * (dN_temp1 + dN_temp2).unsqueeze(dim=1)
            # print(meanN[0]) #print the weight out, you can check most weights are pushing to zero: very sparse model
            
        self.expect_beta_inv = gamma_dN / gamma_cN # (ny,1)
        self.mN = meanN.clone() # (ny,nx)
        self.SN = covarianceN.clone() # (ny,nx,nx)
            
        return self
    
    def marginal_likelihood(self):
        ''' should be a differentiable loss function
            for MLP training
        '''
        return
    
    def predict(self, x, return_var=False):
        mean = x @ self.mN.T
        if return_var:
            var = th.einsum("bij,nj->nib", self.SN, x) # (N,nx,ny)
            var = self.expect_beta_inv.T + th.einsum("ni,nib->nb", x, var) # (N,ny)
            return mean, var
        else:
            return mean
    
    def get_mostuncertain_query(self, x):
        return
    
if __name__ == "__main__":
    # example
    import matplotlib.pyplot as plt
    import numpy as np
    
    X = np.linspace(-10,10,1000)[:,np.newaxis]
    Y = np.concatenate([np.cos(X), np.sin(X), np.cos(X)-np.sin(0.5*X)], axis=1) + np.random.randn(1000,3) * 0.2
    
    from aslearn.feature.polynomial import PolynomialFT
    from aslearn.feature.fourier import FourierFT
    poly = PolynomialFT(degree=2)
    fri = FourierFT(degree=[0.5,1,2,3,4])
    X_f = poly(fri(X))
    print("Feauture dim: ", X_f.shape[1])
    
    X_t, Y_t = th.from_numpy(X_f).to(device).double(), th.from_numpy(Y).to(device).double()
    import time
    s = time.time()
    # we only use the middle 100 data ! but generalize surprisingly well, 
    # because we include the hidden pattern directly in the feature.
    blr = BayesLR().fit(X_t[400:500], Y_t[400:500], iter_num=500)
    e = time.time()
    print(e-s)
    
    
    pred, var = blr.predict(X_t, return_var=True)
    pred, var = pred.detach().cpu().numpy(), var.detach().cpu().numpy()
    plt.figure(figsize=[5,7])
    plt.subplot(311)
    plt.plot(X[:,0], Y[:,0], 'r.')
    plt.plot(X[:,0], pred[:,0], 'b.')
    plt.plot(X[:,0], pred[:,0]+3*var[:,0], 'b-.')
    plt.plot(X[:,0], pred[:,0]-3*var[:,0], 'b-.')
    plt.subplot(312)
    plt.plot(X[:,0], Y[:,1], 'r.')
    plt.plot(X[:,0], pred[:,1], 'b.')
    plt.plot(X[:,0], pred[:,1]+3*var[:,1], 'b-.')
    plt.plot(X[:,0], pred[:,1]-3*var[:,1], 'b-.')
    plt.subplot(313)
    plt.plot(X[:,0], Y[:,2], 'r.')
    plt.plot(X[:,0], pred[:,2], 'b.')
    plt.plot(X[:,0], pred[:,2]+3*var[:,2], 'b-.')
    plt.plot(X[:,0], pred[:,2]-3*var[:,2], 'b-.')
    plt.tight_layout()
    plt.show()
