import torch as th
from aslearn.parametric.regression import Regression

class BayesLR(Regression):
    ''' Fully Bayesian linear regression,
        the hyperparameters are optimized through mean-field variational inference.
        the prior of weight and output noise are assumed to be fully decoupled Gaussian.
        We treat the prior of weight as diagonal Gaussian and the prior of precision as a multiplication of Gamma distributions,
        this leads to an equvalent representation to the variational Relevance Vector Machine (V-RVM) for multivariate regression.
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, Y):
        '''
        '''
        return self
    
    def marginal_likelihood(self):
        '''
        '''
        return 
    
    def predict(self, x):
        return
    
if __name__ == "__main__":
    # example
    import matplotlib.pyplot as plt
    X = th.linspace(-5,5,100)
    X = X.unsqueeze(dim=0).repeat(2,1)
    Y = th.cat([th.cos(X[0:1,:]), th.sin(X[1:2,:])], dim=0) + th.randn(2,100) * 0.1
    
    plt.subplot(211)
    plt.plot(X[0,:], Y[0,:], 'r.')
    plt.subplot(212)
    plt.plot(X[1,:], Y[1,:], 'r.')
    plt.show()