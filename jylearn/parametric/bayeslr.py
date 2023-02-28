import torch as th
from jylearn.parametric.regression import Regression

# TODO
class BayesLR(Regression):
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, Y):
        '''
        '''
        return self
    
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