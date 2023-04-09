import numpy as np
from sklearn.preprocessing import PolynomialFeatures as pf

class PolynomialFT:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly = pf(degree)
        
    def __call__(self, X):
        ''' check if the first entry is 1
            Then conduct the polynomial expansion
        '''
        if np.all( X[:,0]==1. ):
            X = X[:,1:]
        else:
            pass
        return self.poly.fit_transform(X)