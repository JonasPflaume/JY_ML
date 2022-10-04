import numpy as np
from sklearn.preprocessing import PolynomialFeatures

class PolynomialFT:
    def __init__(self, degree=2):
        self.degree = degree
        self.poly = PolynomialFeatures(degree)
        
    def fit_transform(self, X):
        ''' check if the first entry is 1
            Then conduct the polynomial expansion
        '''
        if np.all( X[:,0]==1. ):
            X = X[:,1:]
        else:
            pass
        return self.poly.fit_transform(X)