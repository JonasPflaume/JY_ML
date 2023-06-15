import torch as th
from aslearn.base.regression import Regression
from decimal import Decimal
from numba import jit
import numpy as np
from scipy.optimize import minimize


device = "cuda" if th.cuda.is_available() else "cpu"

class VRVM(Regression):
    ''' Fully Bayesian linear regression,
        the hyperparameters are optimized through mean-field variational inference.
        the prior of weight and output noise are assumed to be fully decoupled Gaussian.
        We treat the prior of weight as diagonal Gaussian and the prior of precision as a multiplication of Gamma distributions,
        this leads to an equvalent representation to the variational Relevance Vector Machine (V-RVM) for multivariate regression.
        TODO: mechanism for removal of the less relevant features
    '''
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.__initialize_parameters(input_dim, output_dim)
        
    def __initialize_parameters(self, nx, ny, noise_init=5, regularization_init=1e-6):
        ''' initialize the variational parameters to make the re-training starts warmly.
        '''
        # initialize model parameter
        # q(alpha): prior of weight precision, product of gamma distributions, 
        # gamma_a:(ny,nx), gamma_b:(ny,nx)
        # initialized to near 0. is usually a good choice
        self.gamma_aN = regularization_init * th.ones(ny, nx).to(device).double()
        self.gamma_bN = regularization_init * th.ones(ny, nx).to(device).double()
        self.gamma_a0 = self.gamma_aN.clone()
        self.gamma_b0 = self.gamma_bN.clone()
        
        # q(beta): prior of precision of output noise
        # gamma_c:(ny,1), gamma_d:(ny,1)
        # initialized to near 0. is usually a good choice
        self.gamma_cN = noise_init * th.ones(ny, 1).to(device).double()
        self.gamma_dN = noise_init * th.ones(ny, 1).to(device).double()
        self.gamma_c0 = self.gamma_cN.clone()
        self.gamma_d0 = self.gamma_dN.clone()
        
        # q(omega): prior of weights
        self.meanN = th.zeros(ny, nx).to(device).double()
        self.covarianceN = th.zeros(ny, nx, nx).to(device).double()
        
    def fit(self, X, Y, tolerance=1e-6, max_inter_num=5000, verbose=2, iterative=True):
        '''
        '''
        # number of data, feature dimension
        # TODO: I doubt the update sequence is wrong!
        N, nx = X.shape
        gamma_aN = self.gamma_aN.clone()
        gamma_bN = self.gamma_bN.clone()
        gamma_cN = self.gamma_cN.clone()
        gamma_dN = self.gamma_dN.clone()
        
        meanN = self.meanN.clone()
        covarianceN = self.covarianceN.clone()

        XTX = th.linalg.matmul(X.T, X)
        XTY = th.linalg.matmul(X.T, Y)
        
        # coordinate decent estimation
        diagonal_index = th.arange(nx)
        stop_flag = False
        if not iterative:
            tolerance = 1.
            
        step_counter = 0
        curr_tolerance_criterion = float("inf")
        while not stop_flag:
            step_counter += 1
            if verbose>1:
                if step_counter % 50 == 0:
                    print("step: ", step_counter, " tolerance: {:.2E}".format(Decimal(curr_tolerance_criterion)))
                else:
                    print("", end=">")
            # expect terms

            E_beta = gamma_cN / gamma_dN # (ny,1)
            E_diag_alpha = gamma_aN / gamma_bN # don't exand this term into diag matrices (ny,nx)

            # update weights q(omega)
            cov_temp = E_beta.unsqueeze(dim=2) * XTX # (ny,nx,nx), utilize the broadcast of torch
            cov_temp[:,diagonal_index,diagonal_index] += E_diag_alpha # add E_diag_alpha to diagonal

            L_cov_temp = th.linalg.cholesky(cov_temp)
            covarianceN = th.cholesky_inverse(L_cov_temp)
            # when meet error "Try to change initial value of each priors
            
            mean_temp = XTY # (nx, ny)
            mean_temp = th.einsum("bij,jb->bi", covarianceN, mean_temp) # (ny,nx)
            mean_temp = E_beta * mean_temp
            mean_change = mean_temp - meanN
            curr_tolerance_criterion = (th.abs( mean_change ).sum() / th.numel( mean_change )).item()
            if curr_tolerance_criterion  < tolerance or step_counter > max_inter_num:
                stop_flag = True

            meanN = mean_temp
            
            if iterative:
            # update q(alpha)
                gamma_aN = self.gamma_a0 + 0.5
                gamma_bN = self.gamma_b0 + 0.5 * covarianceN[:,diagonal_index, diagonal_index] + 0.5 * meanN ** 2.
                
                # update q(beta)
                gamma_cN = self.gamma_c0 + N / 2.0
                
                dN_temp1 = Y.T - th.einsum("ij,bj->bi", X, meanN) # (ny,N)
                dN_temp1 = th.linalg.norm(dN_temp1, dim=1)**2. # (ny,)
                
                dN_temp2 = th.einsum("ij,bjk->bik", XTX, covarianceN) # (ny,nx,nx)
                dN_temp2 = th.einsum("bii->b", dN_temp2) # (ny,) calc trace

                gamma_dN = self.gamma_d0 + 0.5 * (dN_temp1 + dN_temp2).unsqueeze(dim=1)
                # print(meanN[0]) #print the weight out, you can check most weights are pushing to zero: very sparse model
        
        # update global parameters
        self.gamma_aN = gamma_aN
        self.gamma_bN = gamma_bN
        self.gamma_cN = gamma_cN
        self.gamma_dN = gamma_dN
        
        # self.meanN = meanN
        # self.covarianceN = covarianceN
        
        self.expect_beta_inv = gamma_dN / gamma_cN # (ny,1)
        self.mN = meanN # (ny,nx)
        self.SN = covarianceN # (ny,nx,nx)
        if verbose:
            print("coordinate decent completed: ", step_counter, " steps.")
        return self
    
    def marginal_likelihood(self):
        ''' TODO
            No other function, can only be used to score the feature.
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
        
    @staticmethod
    def active_data_inquiry(degree, SN, box_cons, AL_obj, AL_jac, K=10):
        ''' designed for polynomial active learning
        '''
        SN = np.ascontiguousarray(SN)
        x_his = []
        f_his = []
        bounds = []
        for i in range(box_cons.shape[1]):
            bounds.append((box_cons[0,i], box_cons[1,i]))
        bounds = tuple(bounds)
        for _ in range(K):
            # print("Hi!")
            x0 = np.random.uniform(box_cons[0], box_cons[1])
            res_i = minimize(fun=AL_obj, x0=x0, method='L-BFGS-B', jac=AL_jac, args=(degree, SN), bounds=bounds)
            
            x_his.append(res_i.x)
            f_his.append(res_i.fun)
            
        opt_index = np.argmin(f_his)
        x_opt = x_his[opt_index]
        return x_opt


# @jit(nopython=True)
# def polynomialAL_hess(x, degree, SN):
#     x = x.reshape(1,-1)
#     ny = SN.shape[0]

#     poly_J = polynomial_jacobian(x, degree)
#     poly_J = np.ascontiguousarray(poly_J)
#     H = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
#     for i in range(ny):
#         H += - poly_J.T @ SN[i] @ poly_J
    
#     return H
    