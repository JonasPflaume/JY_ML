from aslearn.base.regression import Regression
import torch as th
from torch.optim import LBFGS
import numpy as np
device = "cuda" if th.cuda.is_available() else "cpu"

## TODO:
# 1. kernel class
# 2. rewrite GPR
class ExpectedGPR(Regression):
    
    def __init__(self, kernel):
        '''
            kernel:
                    the kernel function
        '''
        self.kernel = kernel
        self.evidence_evaluate_counter = 0

    def fit(self, X, Y, VY,
            call_hyper_opt=True, 
            lbfgs_evidence_barier=-1e8,
            verbose=False):
        '''
        '''
        self.kernel.start_autograd()
        
        if len(X.shape) == 1:
            # in case the data is 1-d
            X = X.reshape(-1,1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1,1)
        n = X.shape[0]
        m = Y.shape[1] # output dimension
        with th.no_grad():
            self._X = X
        # call the hyperparameter optimization
        
        if call_hyper_opt:
            evidence = float("-inf")
            self.kernel.train()

            optimizer = LBFGS(params=self.kernel.parameters(), 
                                lr=1e-1, 
                                max_iter=200,
                                tolerance_change=1e-6,
                                line_search_fn="strong_wolfe")
            
            self.curr_evidence = evidence
            def closure():
                self.kernel.guarantee_non_neg_param()
                        
                self.evidence_evaluate_counter += 1
                optimizer.zero_grad()
                try:
                    evidence = ExpectedGPR.evidence(self.kernel, X, Y, VY)
                    objective = - evidence
                    objective.backward()
                except:
                    evidence = th.tensor(lbfgs_evidence_barier)
                    objective = -evidence
                self.curr_evidence = evidence.item()
                if verbose:
                    if self.evidence_evaluate_counter % 50 == 0:
                        print("Current evidence: %.2f" % evidence.item())
                    else:
                        print(">", end="")
                return objective
            
            optimizer.step(closure)
            self.kernel.guarantee_non_neg_param()
                
            self.kernel.eval()
            self.kernel.stop_autograd()
        # output guarantee for kernel postive param
        self.kernel.guarantee_non_neg_param()
        K = self.kernel(X, X) # K has shape (ny, n, n)
        try:
            u = th.linalg.cholesky(K) # u has shape (ny, n, n)
        except:
            if verbose:
                print("The cho_factor meet singular matrix, now add damping...")
            K += th.eye(K.shape[1]).unsqueeze(0).repeat(m, 1, 1).to(device) * 1e-6
            u = th.linalg.cholesky(K) # u has shape (ny, n, n)
        
        Y = Y.permute(1, 0)
        Y = Y.unsqueeze(2)
        self.alpha = th.cholesky_solve(Y, u) # (ny, n, 1) solve ny linear system independently
        self.L = u # shape (ny, n, n)
        
        white_diag = self.kernel.white_diag(X)
        diag_index = th.arange(len(X))
        K[:, diag_index, diag_index] = K[:, diag_index, diag_index] - white_diag
        self.posterior = th.einsum("bij,bjk->bik", K, self.alpha)
        return self
        
    @staticmethod
    def evidence(kernel, X, Y, VY, channel_process="sum"):
        n = X.shape[0]
        m = Y.shape[1] # output dimension

        
        Sc = VY.sum(dim=0)
        white_noise = kernel.white_diag(X[0:1,:]).squeeze(dim=0)
        Sc = - 1/2 * (Sc/white_noise)
        
        K = kernel(X, X) # K has shape (ny, n, n)
        try:
            u = th.linalg.cholesky(K) # u has shape (ny, n, n)
        except:
            K = K + th.eye(K.shape[1]).unsqueeze(0).repeat(m, 1, 1).to(device) * 1e-6
            u = th.linalg.cholesky(K) # u has shape (ny, n, n)
            
        Y = Y.permute(1, 0)
        Y = Y.unsqueeze(2)

        alpha = th.cholesky_solve(Y, u) # (ny, n, 1) solve ny linear system independently
        L = u # shape (ny, n, n)
        
        # print the margianl likelyhood
        # Y (ny, n, 1), alpha (ny, n, 1), L (ny, n, n)
        diagonal_term = th.log(th.diagonal(L, dim1=1, dim2=2)).sum(dim=1) # (ny,)
        evidence = - 0.5 * th.einsum("ijk,ijk->i", Y, alpha) - diagonal_term - n / 2 * th.log(2*th.tensor([th.pi]).to(device))
        evidence = evidence + Sc
        if channel_process == "sum":
            evidence = evidence.sum(axis=-1)
        elif channel_process == "mean":
            evidence = evidence.mean(axis=-1)

        return evidence
    
    def predict(self, x, return_var=False):
        '''
        '''
        if len(x.shape) == 1:
            # in case the data is 1-d
            x = x.reshape(-1,1)
            
        k = self.kernel(x, self._X) # (ny, n_*, n), alpha: (ny, n, 1)
        mean = th.einsum("ijk,ikb->ji", k, self.alpha)
        if return_var:
            k = k.permute(0,2,1)
            v = th.linalg.solve_triangular(self.L, k, upper=False) # (ny, n, n_*)
            v = v.permute(2, 0, 1) # (n_*, ny, n)
            prior_std = self.kernel(x,x,diag=True) # the diag in kernel base class changed. (ny, n_*)
            var = prior_std.T - th.einsum("ijk,ijk->ij", v, v)
            white_noise = self.kernel.white_diag(x).T
            var += white_noise
            del k, v, prior_std
            th.cuda.empty_cache()
            return mean, var.reshape(-1,self.kernel.output_dim)
        else:
            del k
            th.cuda.empty_cache()
            return 
        
if __name__ == "__main__":
    # test the gp
    X = th.linspace(-5,5,40).reshape(-1,1).to(device).double()
    Y = th.sin(X) + th.randn_like(X).to(device).double() * 0.2
    VY = th.zeros_like(X) * 1e-2
    
    from aslearn.kernel.kernels import RBF, White
    l = np.ones([1,1]) * 1
    c = np.ones([1,]) * 0.2
    kernel = RBF(l=l, dim_in=1, dim_out=1) + White(c=c, dim_in=1, dim_out=1)
    gpr = ExpectedGPR(kernel=kernel).fit(X, Y, VY, verbose=True)
    
    pred, var = gpr.predict(X, return_var=True)
    
    pred, var = pred.detach().cpu().numpy(), var.detach().cpu().numpy()
    print(kernel)
    print(var[0])
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    Y_pos = gpr.posterior.squeeze(dim=0).detach().cpu().numpy()
    import matplotlib.pyplot as plt
    plt.plot(X, Y_pos, '-b')
    # plt.plot(X, pred+var, '-.b')
    # plt.plot(X, pred-var, '-.b')
    plt.plot(X, Y, 'r.')
    plt.show()
    pass