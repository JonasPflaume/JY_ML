import numpy as np
from scipy.spatial.distance import cdist
import casadi as cs

class BellCurve:
    def __init__(self, degree=2, l=1.):
        self.degree = degree
        self.l = l
        
    def fit(self, X):
        self.dtype = X.dtype
        assert len(X) >= self.degree, "You shouldn't choose more cluster than the dataset"
        X_sub_index = np.random.choice(np.arange(len(X)), self.degree, replace=False)
        self.X_sub = X[X_sub_index]
        return self
        
    def set(self, X_sub):
        self.dtype = X_sub.dtype
        self.X_sub = X_sub
        
    def __call__(self, X, casadi=False):
        '''
        '''
        if casadi:
            # only designed for X==(nx,)
            bell_values = [cs.SX(1), X]
            for i in range(len(self.X_sub)):
                bell_value = cs.exp( - cs.norm_2(X/self.l-self.X_sub[i]/self.l) ** 2 / 2)
                bell_values.append(bell_value)
            X_aug = cs.vcat(bell_values)
        else:
            X_aug = np.concatenate([np.ones([X.shape[0],1], dtype='f'), X], axis=1)
            
            dist = cdist(X/self.l, self.X_sub/self.l) # (n, degree)
            feature = np.exp( - dist ** 2 / 2)

            X_aug = np.concatenate( [X_aug, feature], axis=1)        
            X_aug = X_aug.astype(self.dtype)
        return X_aug
