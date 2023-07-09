from aslearn.kernel.kernels import Kernel
from aslearn.base.regression import Regression
import torch as th
from torch.optim import LBFGS
device = "cuda" if th.cuda.is_available() else "cpu"

from aslearn.common_utils.check import  HAS_METHOD,RIGHT_SHAPE,WARNING,REPORT_VALUE,PRINT

class ExactGPR(Regression):
    ''' Exact Gaussian process regressor
        kernel: input kernel instance
        
        - the hyperparameters of the kernel is optimized through maximum marginal likelihood
        - get_params and set_params were designed for further application e.g. MCMC.
        - I assume the likelihood is Gaussian.
        - mean prior can be appointed through a callable object with predict() method.
    '''
    def __init__(self, kernel:Kernel) -> None:
        super().__init__()
        self.kernel = kernel
        
    def fit(self, X:th.Tensor, Y:th.Tensor, call_opt=True, mean_prior=None, info_level=0, max_iter=2000, **evidence_inputs):
        ''' infor_level = 0, print nothing
                        = 1, print normal stuff
                        >= 2, print warning
        '''
        self.kernel.train()
        
        nx = self.kernel.input_dim
        ny = self.kernel.output_dim
        with th.no_grad():
            self._X = X
            self._Y = Y
            
        N = len(X)
        RIGHT_SHAPE(X, (N, nx))
        RIGHT_SHAPE(Y, (N, ny))
        if mean_prior != None:
            HAS_METHOD(mean_prior, "predict")
            # store it for prediction
            self.mean_prior = mean_prior
        else:
            self.mean_prior = None
            
        self.curr_loss = float("inf")
        if call_opt:
            self.evidence_eval_time = 0
            optimizer = LBFGS(params=self.kernel.parameters(), lr=1., max_iter=max_iter, line_search_fn="strong_wolfe")
            
            def lbfgs_closure():
                self.evidence_eval_time += 1
                # is_clamp_triggered = self.kernel.guarantee_non_neg_params()
                # WARNING(is_clamp_triggered, "Some parameters hit the lower bound.", info_level)
                optimizer.zero_grad()
                elbo = self.evidence(X, Y, self.mean_prior, **evidence_inputs)
                loss = - elbo
                loss.backward()
                self.curr_loss = loss.item()
                
                if self.evidence_eval_time % 50 == 0:
                    REPORT_VALUE(self.curr_loss, "Curr loss:", info_level)
                else:
                    PRINT(">", info_level)
                return loss
            
            optimizer.step(lbfgs_closure)
        REPORT_VALUE(self.curr_loss, "Final loss:", info_level)
        
        # self.kernel.guarantee_non_neg_params()
        K = self.kernel(X, X)
        noise = self.kernel.noise(X, X) # (ny,1)
        diagonal_index = th.arange(N)
        K[:,diagonal_index,diagonal_index] += noise
        
        try:
            L = th.linalg.cholesky(K)
        except:
            # add a diagonal small number
            try:
                L = th.linalg.cholesky(K + th.eye(len(X)).unsqueeze(dim=0).to(device).double() * 1e-5)
            except:
                raise ValueError("Still not PSD, you may change the initial hyperparameters...")
        
        # (K + Q)^{-1} (f-u)
        if self.mean_prior != None:
            u = self.mean_prior.predict(X)
            RIGHT_SHAPE(u, Y.shape)
        else:
            u = th.zeros_like(Y) # (N, ny)
            
        self.L = L # (ny, N, N)
        target = (Y-u).permute(1,0).unsqueeze(dim=2) # (ny, N, 1)
        RIGHT_SHAPE(target, (ny, N, 1))
        RIGHT_SHAPE(L, (ny, N, N))
        self.alpha = th.cholesky_solve(target, L) # (ny, N, 1)
        self.kernel.eval()
        
        return self
    
    def posterior_mean(self,) -> th.Tensor:
        ''' this method provide the posterior mean of the GPR
            the autograd is blocked !
        '''
        with th.no_grad():
            if self.mean_prior != None:
                u = self.mean_prior.predict(self._X)
                RIGHT_SHAPE(u, self._Y.shape)
            else:
                u = th.zeros_like(self._Y).double().to(device) # (N, ny)

            K = self.kernel(self._X, self._X)
            Ef = th.einsum("bij,bjk->bik", K, self.alpha) + u.permute(1, 0).unsqueeze(dim=2) # (ny, N, 1)
            Ef = Ef.squeeze(dim=2).permute(1, 0)

        return Ef # (N, ny)
    
    def posterior_cov(self,) -> th.Tensor:
        ''' this method provide the posterior covariance of the GPR
            the autograd is blocked !
        '''
        with th.no_grad():

            K = self.kernel(X, X)
                
            temp = th.cholesky_solve(K, self.L)
            Cov_f = K - th.einsum("bij,bjk->bik", K, temp)

        return Cov_f # (ny, N, N)
    
    def predict(self, x:th.Tensor, return_std=False) -> th.Tensor:
        ''' predict the unseen data
        '''
        RIGHT_SHAPE(x, (-1, self.kernel.input_dim))
        N = len(x)
        with th.no_grad():
            if self.mean_prior != None:
                u_x = self.mean_prior.predict(x)
                RIGHT_SHAPE(u_x, (-1, self.kernel.output_dim))
            else:
                u_x = th.zeros(len(x), self.kernel.output_dim).double().to(device) # (N, ny)

            k = self.kernel(x, self._X) # (ny, Ntest, Ntrain)
            Ef = th.einsum("bij,bjk->bik", k, self.alpha) + u_x.permute(1, 0).unsqueeze(dim=2) # (ny, Ntest, 1) alpha:(ny, Ntrain, 1)
            Ef = Ef.squeeze(dim=2).permute(1, 0)

            if return_std:
                kxx = th.diagonal(self.kernel(x, x), dim1=1, dim2=2) # (ny, Ntest)
                temp_term = th.cholesky_solve(k.permute(0,2,1), self.L)
                var_f = th.einsum("bij,bji->bi", k, temp_term) # (ny, Ntest)
                noise_f = self.kernel.noise(x, x).repeat(1, N) # (ny,Ntest)
                var_f = (noise_f + kxx - var_f).permute(1, 0)
                return Ef, th.sqrt(var_f) # sqrt variance to std
            return Ef
    
    def gpr_evidence(self, X:th.Tensor, Y:th.Tensor, mean_prior=None) -> th.Tensor:
        ''' marginal likelihood of the GPR
        '''
        RIGHT_SHAPE(X, (-1,-1))
        RIGHT_SHAPE(Y, (-1,-1))
        
        if mean_prior != None:
            u = mean_prior.predict(X)
            RIGHT_SHAPE(u, Y.shape)
        else:
            u = th.zeros_like(Y) # (N, ny)
            
        N = len(Y)
        K = self.kernel(X, X)
        ny = Y.shape[1]
        noise = self.kernel.noise(X, X) # (ny,1)
        diagonal_index = th.arange(N)
        K[:,diagonal_index,diagonal_index] += noise
        try:
            L = th.linalg.cholesky(K)
        except:
            # add a diagonal small number
            try:
                L = th.linalg.cholesky(K + th.eye(len(X)).unsqueeze(dim=0).to(device).double() * 1e-5)
            except:
                raise ValueError("Still not PSD, you may change the initial hyperparameters...")
        
        # log det(LL^T) = 2 sum_i log (L_ii)
        det_term = - th.log(th.diagonal(L, dim1=1, dim2=2)).sum(dim=1)
        RIGHT_SHAPE(det_term, (ny,))
        
        Y = Y - u # prior mean
        Y = Y.permute(1,0).unsqueeze(dim=2) # (ny,N,1)
        quadratic_term = th.cholesky_solve(Y, L)
        quadratic_term = th.einsum("bij,bjk->b", Y.transpose(dim0=2,dim1=1), quadratic_term) # (ny,)
        quadratic_term = - 0.5 * quadratic_term
        pi_term = - N / 2 * th.log(th.tensor(2*th.pi))
        elbo = (quadratic_term + det_term + pi_term).sum(dim=0)
        
        return elbo
    
    def evidence(self, X:th.Tensor, Y:th.Tensor, mean_prior=None, **evidence_inputs) -> th.Tensor:
        ''' this method can be overwritten to change the training evidence function for gpr
            in this vanilla implementation, it will just return the normal gpr evidence
        '''
        return self.gpr_evidence(X, Y, mean_prior)
    
    def get_params(self,):
        ''' no matter which stage the gpr is, 
            we first stop the autograd to avoid numerical error.
            As long as you call this method, 
            which means you won't need the autograd until you call the start_autograd by hand.
        '''
        self.kernel.stop_autograd()
    
    def set_params(self,):
        ''' no matter which stage the gpr is, 
            we first stop the autograd to avoid numerical error.
            As long as you call this method, 
            which means you won't need the autograd until you call the start_autograd by hand.
        '''
        self.kernel.stop_autograd()
        
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    ### test
    # test the evidence function
    print("------- Test evidence -------")
    from aslearn.kernel.kernels import RBF, White
    X = th.randn(10,5).double().to(device)
    Y = th.randn(10,2).double().to(device)
    
    kernel = RBF(5,2) + White(5,2)
    gpr = ExactGPR(kernel=kernel)
    loss = gpr.evidence(X, Y)
    try:
        loss.backward()
    except:
        raise ValueError
    
    # give a mean_prior
    from aslearn.parametric.mllr import MLLR
    mean_prior = MLLR().fit(X, Y, info_level=1)
    kernel = RBF(5,2) + White(5,2)
    gpr = ExactGPR(kernel=kernel)
    loss = gpr.evidence(X, Y, mean_prior=mean_prior)
    # with small dataset, with a mean_prior imposes a strong assumption, 
    # usually the elbo will be larger than 0 mean.
    try:
        loss.backward()
    except:
        raise ValueError
    print("PASS the evidence tests")
    
    # fit noise function, check the number of white noise
    X = th.linspace(-2,4,50).reshape(-1,1).double().to(device)
    Y = th.sin(X) + th.randn_like(X) * 0.3
    
    kernel = RBF(1,1) + White(1,1)
    
    # check the mean_prior and neural network learning
    mp = MLLR().fit(X, Y, info_level=1)
    
    # you can input mp as a mean prior
    gpr = ExactGPR(kernel=kernel).fit(X, Y, mean_prior=None, info_level=1) # with piror or without prior
    print(kernel) # looks good
    # calc the posterior
    mean = gpr.posterior_mean()
    cov = gpr.posterior_cov() # (ny, N, N)
    cov = th.diagonal(cov, dim1=1, dim2=2).T
    mean = mean.detach().cpu().numpy()
    cov = cov.detach().cpu().numpy()
    plt.plot(mean, '-b')
    plt.plot(mean+cov, '-.b')
    plt.plot(mean-cov, '-.b')
    plt.plot(Y.detach().cpu().numpy(), '.r')
    plt.title("GP posterior")
    plt.show() # looks good
    plt.close()
    
    # test the predict
    X_test=th.linspace(-2,10,100).reshape(-1,1).double().to(device)
    Y_test=th.sin(X_test)
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    kernel_ = RBF() + WhiteKernel()
    gpr_ = GaussianProcessRegressor(kernel=kernel_).fit(X.detach().cpu().numpy(), Y.detach().cpu().numpy())
    pred_, std_ = gpr_.predict(X_test.detach().cpu().numpy(), return_std=True)
    std_ = std_.reshape(-1,1)
    
    pred, std = gpr.predict(X_test, return_std=True)
    pred = pred.detach().cpu().numpy()
    std = std.detach().cpu().numpy()
    plt.plot(pred, '-b')
    plt.plot(pred+1.96*std, '-.b')
    plt.plot(pred-1.96*std, '-.b')
    plt.plot(pred_, '-c')
    plt.plot(pred_+1.96*std_, '-.c')
    plt.plot(pred_-1.96*std_, '-.c')
    plt.plot(Y_test.detach().cpu().numpy(), '.r')
    plt.title("Test GP prediction (with linear regression prior)")
    plt.show()
    plt.close()
    # looks good
    # test the get/set_params (for future, don't need now)
    pass