import numpy as np
# from aslearn.feature.features_backend import feature_polynomial
from sklearn.preprocessing import PolynomialFeatures
from abc import ABC

class Feature(ABC):
    """ feature base class
        enables concatenation, composition of derived classes 
        and generation of their Jacobian calculation
    """
    # NOTE:1 remember the jacobian should be a static jit function !
    # NOTE:2 enable enables concatenation, composition
    # NOTE:3 enable feature pop out, designed for sparse model! jacobian should be correct.
    
class FourierFT(Feature):
    def __init__(self, degree):
        ''' degree is the phase factor of trigonometric funcs.
        '''
        self.degree = degree
        
    def __call__(self, X):
        ''' check if the first entry is 1
            Then conduct the sin cos expansion
        '''
        if np.all( X[:,0]==1. ):
            X_aug = X.copy()
            X = X[:,1:]
        else:
            X_aug = np.concatenate([np.ones([X.shape[0],1], dtype='f'), X], axis=1)
        
        for d in self.degree:
            X_aug = np.concatenate([X_aug, np.cos(d*X)], axis=1)
            X_aug = np.concatenate([X_aug, np.sin(d*X)], axis=1)
            
        return X_aug

class PolynomialFT(Feature):
    ''' Bad implmentation, much slower than sklearn
    '''
    def __init__(self, degree=2):
        self.degree = degree
        self.poly = PolynomialFeatures(degree=degree)
        
    def __call__(self, X):
        return self.poly.fit_transform(X)
    
class SquareWaveFT(Feature):
    def __init__(self, frequencies):
        ''' square wave filtering
        '''
        self.frequencies = frequencies
        
    def __call__(self, X):
        '''
        '''
        X_aug = X.copy()
        for freq in self.frequencies:
            X_aug = np.concatenate([X_aug, np.sign(np.sin(2*np.pi*freq*X))], axis=1)
        return X_aug