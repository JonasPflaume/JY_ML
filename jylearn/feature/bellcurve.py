import numpy as np
from scipy.spatial.distance import cdist

class BellCurve:
    def __init__(self, degree=2, l=1.):
        self.degree = degree
        self.l = l
        
    def fit(self, X):
        self.dtype = X.dtype
        assert len(X) > self.degree, "You shouldn't choose more cluster than the dataset"
        X_sub_index = np.random.choice(np.arange(len(X)), self.degree, replace=False)
        self.X_sub = X[X_sub_index]
        
    def __call__(self, X):
        ''' check if the first entry is 1
        '''
        if np.all( X[:,0]==1. ):
            X_aug = X.copy()
            X = X[:,1:]
        else:
            X_aug = np.concatenate([np.ones([X.shape[0],1], dtype='f'), X], axis=1)
        
        dist = cdist(X/self.l, self.X_sub/self.l) # (n, degree)
        feature = np.exp( - dist ** 2 )

        X_aug = np.concatenate( [X_aug, feature], axis=1)        
        X_aug = X_aug.astype(self.dtype)
        return X_aug
