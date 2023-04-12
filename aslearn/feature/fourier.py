import numpy as np

class FourierFT:
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