from aslearn.kernel.kernels import Kernel
from aslearn.base.regression import Regression
import torch as th
from torch.optim import LBFGS

from aslearn.common_utils.check import has_method, right_shape

class ExactGPR(Regression):
    ''' Exact Gaussian process regressor
        kernel: input kernel instance
        
        - the hyperparameters of the kernel is optimized through maximum marginal likelihood
        - get_params and set_params were designed for further application e.g. MCMC.
        - I assume the likelihood is Gaussian.
        - mean prior can be appointed through a callable mean function
    '''
    def __init__(self, kernel:Kernel) -> None:
        super().__init__()
        self.kernel = kernel
        
    def fit(self, X:th.Tensor, Y:th.Tensor, call_opt=True, mean_prior=None):
        # gp posterior
        nx = self.kernel.input_dim
        ny = self.kernel.output_dim
        N = len(X)
        right_shape(X, (N, nx))
        right_shape(Y, (N, ny))
        if mean_prior != None:
            has_method(mean_prior, "predict")
            
        return self
    
    def predict(self, x:th.Tensor, return_std=False):
        pass
    
    def evidence(self, X:th.Tensor, Y:th.Tensor, mean_prior=None):
        ''' marginal likelihood of the GPR
        '''
        pass
    
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
    ### test
    
    # fit noise function, check the number of white noise
    
    # given the mean function
    
    # test the get/set_params
    
    pass