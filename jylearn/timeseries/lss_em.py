import torch as th
import torch.nn as nn
from torch.optim import Adam
import numpy as np
th.set_printoptions(precision=3)

##
##  comments:
##              1. EM LSS will overfit when dim_x is large, or trained with small dataset
##              2. The initialization of parameters play a very important role for numerical stability!
##                 When convergence problem or bad result happens, run fit again may happens to start training with a good initialization.
##              3. Cholesky smoother can be implemented through tensor parallelisation to avoid computing cholesky factor of large sparse matrix
##              4. If you saw the smoothing result is a fold-line connecting the observation points, 
##                 which means the smoother was totally not trusting learned dynamic model.
##

## TODO: evaluate the smoothing variance ?

def extract_diag_block(X_cov, dim_x):
    container = th.zeros(X_cov.shape[0]//dim_x, dim_x, dim_x)
    for i in range(X_cov.shape[0]//dim_x):
        container[i,:,:] = th.linalg.inv(X_cov[i*dim_x:(i+1)*dim_x, i*dim_x:(i+1)*dim_x])
    return container
    
class LSS_Param(nn.Module):
    def __init__(self, dim_x, dim_u, dim_obs):
        super().__init__()
        ## give it a reasonable initialization ##
        
        A = th.randn(dim_x, dim_x)*1e-5 + th.eye(dim_x)
        B = th.randn(dim_x, dim_u)*1e-5
        C = th.randn(dim_obs, dim_x)*1e-5
        
        Gamma_L = th.abs(th.randn(dim_x)) * 5
        K_L = th.abs(th.randn(dim_obs)) * 5

        self.A = nn.parameter.Parameter(A)
        self.B = nn.parameter.Parameter(B)
        self.C = nn.parameter.Parameter(C)
        self.Gamma = nn.parameter.Parameter(Gamma_L) # process noise - log precision matrix
        self.K = nn.parameter.Parameter(K_L) # observation noise
        
        # initial state
        P0_prior_L = th.abs(th.randn(dim_x)) * 5
        x0_prior = th.randn(dim_x,1)
        
        self.P0_prior = nn.parameter.Parameter(P0_prior_L)
        self.x0_prior = nn.parameter.Parameter(x0_prior)

class LSS(object):
    ''' learning linear state space model by em algorithm
        Implemented with pytorch;
    '''
    def __init__(self, LSS_Param):
        self.LSS_Param = LSS_Param
        
    def fit(self, X, U):
        optimizer = Adam(params=self.LSS_Param.parameters(), lr=5e-3)
        self.curr_loss = 0.
        
        dim_x = self.LSS_Param.A.shape[0]
        outloop_history = [float("inf")]
        for _ in range(50):
            with th.no_grad():
                X_filtered, X_smoothed, X_cov = LSS.cholesky_smoothing(self.LSS_Param, X, U) # calc the belief
            for _ in range(100): # no need to centering, this number was hand tuned
                optimizer.zero_grad()
                objective = LSS.joint_likelihood(self.LSS_Param, dim_x, X_smoothed, X, U)
                objective.backward()
                optimizer.step()
                self.curr_loss = objective.item()
            print("Current ELBO: ", self.curr_loss)
            outloop_history.append(self.curr_loss)

        with th.no_grad():
            X_var = extract_diag_block(X_cov, dim_x)
            Y_var_temp = th.einsum("ij,ljk->lik", self.LSS_Param.C, X_var)
            Y_var = th.einsum("lik,kn->lin", Y_var_temp, self.LSS_Param.C.T) + self.LSS_Param.K.unsqueeze(dim=0)
            Y_var = th.diagonal(Y_var, dim1=1, dim2=2)

            X_smoothed = X_smoothed.reshape(-1, dim_x)
            Y_smoothed = th.einsum("ij,lj->li", self.LSS_Param.C, X_smoothed)
            
            X_filtered = X_filtered.reshape(-1, dim_x)
            X_filtered = th.einsum("ij,lj->li", self.LSS_Param.C, X_filtered)

        return X_filtered, Y_smoothed, Y_var
    
    @staticmethod
    def cholesky_smoothing(LSS_Param, X, U):
        ### check step 1
        x0_prior = LSS_Param.x0_prior
        v = th.einsum("ij,bj->bi", LSS_Param.B, U).reshape(-1,1)
        z = th.cat([x0_prior.reshape(-1,1), v, X.reshape(-1,1)], dim=0)
        ###
        
        length, x_dim = len(X), LSS_Param.A.shape[0]
        
        ### check step 2
        H1 = th.eye(length * x_dim)
        H2_temp1 = th.block_diag(*[-LSS_Param.A]*(length-1))
        H2_temp2 = th.cat([th.zeros(x_dim, x_dim*(length-1)), H2_temp1], dim=0)
        H2 = th.cat([H2_temp2, th.zeros(x_dim*length, x_dim)], dim=1)
        H3 = th.block_diag(*([LSS_Param.C]*length))
        H = th.cat([H1 + H2, H3], dim=0)
        ###
        
        ### check step 3
        P0_prior_precision = th.exp(LSS_Param.P0_prior)
        Gamma_precision = th.exp(LSS_Param.Gamma)
        K_precision = th.exp(LSS_Param.K)
        
        W_diag_list = [th.diag(P0_prior_precision)] +\
            [th.diag(Gamma_precision)]*(length-1) + [th.diag(K_precision)]*length
        Winv = th.block_diag(*W_diag_list)
        ### 
        cholesky_term = H.T @ Winv @ H
        try:
            cho_Low = th.linalg.cholesky(cholesky_term)
        except:
            raise ValueError("Please try to train it again, we met bad initial parameters...")

        rhs_term = H.T @ Winv @ z
        
        X_filtered = th.linalg.solve_triangular(cho_Low, rhs_term, upper=False)
        X_smoothed = th.linalg.solve_triangular(cho_Low.T, X_filtered, upper=True)

        return X_filtered, X_smoothed, cholesky_term # (mean, covariance)
    
    @staticmethod
    def joint_likelihood(lss_param, dim_x, X_smoothed, X_obs, U):
        # X = [x0,x1,x2,x3,...,xK]
        # The exact log likelihood was derived to avoid using the pytorch distribution class,
        # the compute time can thus largely reduced.
        
        X_smoothed = X_smoothed.reshape(-1, dim_x)
        nll = 0
        
        ### check step 1
        mean_1 = lss_param.x0_prior.T
        P0_prior_precision = th.exp(lss_param.P0_prior)
        
        x0_rvar = X_smoothed[0].unsqueeze(dim=0) # random variable
        temp1 = th.einsum("j,lj->lj", th.sqrt(P0_prior_precision), x0_rvar - mean_1)
        temp2 = 0.5 * th.log(th.prod(P0_prior_precision)) - 0.5 * th.norm(temp1, dim=1, keepdim=True) ** 2.
        nll -= len(X_smoothed)*temp2.sum()
        # because the magnitude of initial state loss will generally len(X_smoothed) times smaller than other 2 terms
        ###
        
        ### check step 2
        mean_2_ = th.einsum("ij,lj->li", lss_param.A, X_smoothed[:-1])
        mean_2 = mean_2_ + th.einsum("ij,lj->li", lss_param.B, U)
        Gamma_precision = th.exp(lss_param.Gamma)
        
        x_rvar = X_smoothed[1:] # random variable
        temp3 = th.einsum("j,lj->lj", th.sqrt(Gamma_precision), x_rvar - mean_2)
        temp4 = 0.5 * th.log(th.prod(Gamma_precision)) - 0.5 * th.norm(temp3, dim=1, keepdim=True) ** 2.
        nll -= temp4.sum()
        ###
        
        ### check step 3
        mean_3 = th.einsum("ij,lj->li", lss_param.C, X_smoothed)
        K_precision = th.exp(lss_param.K)

        x_rvar_2 = X_obs # random variable
        temp5 = th.einsum("j,lj->lj", th.sqrt(K_precision), x_rvar_2 - mean_3)
        temp6 = 0.5 * th.log(th.prod(K_precision)) - 0.5 * th.norm(temp5, dim=1, keepdim=True) ** 2.
        nll -= temp6.sum()
        ###

        return nll
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from jycontrol.system import Pendulum
    from jylearn.timeseries.utils import collect_rollouts
    
    p = Pendulum()
    X_l, U_l = collect_rollouts(p, 1, 200)
    X, U = X_l[0], U_l[0]
    X_noise, U = th.from_numpy(X + 2/np.max(X)*np.random.randn(*X.shape)).float(), th.from_numpy(U_l[0]).float()
            
    lss_param = LSS_Param(dim_x=4, dim_u=1, dim_obs=2)
    lss = LSS(LSS_Param=lss_param)
    X_filtered, X_smoothed, variance = lss.fit(X_noise, U)


    X_smoothed = X_smoothed.detach().numpy()
    X_smoothed = X_smoothed.reshape(-1, 2)
    
    X_filtered = X_filtered.detach().numpy()
    X_filtered = X_filtered.reshape(-1, 2)
    
    variance = variance.detach().numpy()
            
    plt.subplot(211)
    plt.plot(X_noise[:,0], ".r", label="observation")
    plt.plot(X[:,0], "-b", label="gt")
    plt.plot(X_smoothed[:,0], "-c", label="smoothed")
    plt.plot(X_smoothed[:,0]+variance[:,0], "-.c")
    plt.plot(X_smoothed[:,0]-variance[:,0], "-.c", label="variance")

    plt.grid()
    plt.subplot(212)
    plt.plot(X_noise[:,1], ".r", label="observation")
    plt.plot(X[:,1], "-b", label="gt")
    plt.plot(X_smoothed[:,1], "-c", label="smoothed")
    plt.plot(X_smoothed[:,1]+variance[:,1], "-.c")
    plt.plot(X_smoothed[:,1]-variance[:,1], "-.c", label="variance")

    plt.grid()
    plt.legend()
    plt.show()