import torch as th
import torch.nn as nn
from torch.optim import Adam
import numpy as np
th.set_printoptions(precision=2)


##
##  comments:
##              1. EM LSS will overfit when dim_x is large, or trained with small dataset
##              2. The initialization of parameters play a very important role for numerical stability!
##              3. Cholesky smoother can be implemented through tensor parallelisation to avoid computing cholesky factor of large sparse matrix
##

class LSS_Param(nn.Module):
    def __init__(self, dim_x, dim_u, dim_obs):
        super().__init__()
        ## give it a reasonable initialization ##
        
        A = th.randn(dim_x, dim_x)*1e-4 + th.eye(dim_x)
        B = th.randn(dim_x, dim_u)*1e-4
        C = th.randn(dim_obs, dim_x)*1e-4
        
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
        optimizer = Adam(params=self.LSS_Param.parameters(), lr=1e-2)
        self.curr_loss = 0.
        
        dim_x = self.LSS_Param.A.shape[0]
        outloop_history = [float("inf")]
        while True:
            with th.no_grad():
                X_filtered, X_smoothed, X_cov = LSS.cholesky_smoothing(self.LSS_Param, X, U) # calc the belief

            for _ in range(120): # no need to centering, this number was hand tuned
                optimizer.zero_grad()
                objective = LSS.joint_likelihood(self.LSS_Param, dim_x, X_smoothed, X, U)
                objective.backward()
                optimizer.step()
                self.curr_loss = objective.item()

            print("Current ELBO: ", self.curr_loss)
            outloop_history.append(self.curr_loss)
            if outloop_history[-2] - outloop_history[-1] <= 1e-4:
                break
        with th.no_grad():
            X_cov = th.diag(X_cov)
            X_cov = X_cov.reshape(-1, dim_x)
            X_smoothed = X_smoothed.reshape(-1, dim_x)
            Y_smoothed = th.einsum("ij,lj->li", self.LSS_Param.C, X_smoothed)
            
            X_filtered = X_filtered.reshape(-1, dim_x)
            X_filtered = th.einsum("ij,lj->li", self.LSS_Param.C, X_filtered)
        return X_filtered, Y_smoothed, X_cov
    
    def predict(self, x):
        pass
    
    def traj_predict(self, x, U):
        pass
    
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

        cho_Low = th.linalg.cholesky(cholesky_term)

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
    ### Toy example A = diag([0.2,0.7]) B = [[0.5],[1]], process_noise = N(0, diag([0.03, 0.03]))###
    import matplotlib.pyplot as plt

    with th.no_grad():
        A, B = th.eye(2), th.tensor([[0.5],[1]])
        A[0,0] *= 0.7
        A[0,1] += -0.4
        A[1,1] *= 0.5
        
        ## we need to trade off model dimension and data number!
        time_step = 300
        t = th.linspace(0, 20, time_step)
        U = (th.sin(2*t) + th.sin(0.5*t) + th.cos(t) + th.cos(0.5*t)).unsqueeze(dim=1) * 0.1
        x0 = th.tensor([[0.], [0.]])
        X = th.zeros(time_step+1,2)
        gt_res = th.zeros(time_step+1,2)
        X[0,:] = (x0 + th.randn(2,1)*0.05).squeeze()
        gt_res[0,:] = x0.squeeze()
        step = 0
        # C = I
        for u in U:
            step += 1
            x0 = A@x0 + B@u.unsqueeze(dim=1) # add process noise
            X[step,:] = (x0 + th.randn(2,1)*0.05).squeeze()
            gt_res[step,:] = x0.squeeze()
            
    lss_param = LSS_Param(dim_x=3, dim_u=1, dim_obs=2)
    
    lss = LSS(LSS_Param=lss_param)
    
    X_filtered, X_smoothed, X_cov = lss.fit(X, U)

    X_smoothed = X_smoothed.detach().numpy()
    # print(X_smoothed)
    X_smoothed = X_smoothed.reshape(-1, 2)
    
    X_filtered = X_filtered.detach().numpy()
    # print(X_smoothed)
    X_filtered = X_filtered.reshape(-1, 2)
            
    plt.subplot(211)
    plt.plot(X[:,0], ".r", label="observation")
    plt.plot(gt_res[:,0], "-b", label="gt")
    plt.plot(X_smoothed[:,0], "-c", label="smoothed")
    # plt.plot(X_filtered[:,0], "-k", label="filtered")
    plt.grid()
    plt.subplot(212)
    plt.plot(X[:,1], ".r", label="observation")
    plt.plot(gt_res[:,1], "-b", label="gt")
    plt.plot(X_smoothed[:,1], "-c", label="smoothed")
    # plt.plot(X_filtered[:,1], "-k", label="filtered")
    plt.grid()
    plt.legend()
    plt.show()