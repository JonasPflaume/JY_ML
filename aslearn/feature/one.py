import numpy as np
from aslearn.feature.jcfeatures import feature_one

class OneFT:
    def __init__(self,):
        ''' degree is the phase factor of trigonometric funcs.
        '''
        
    def __call__(self, X):
        return feature_one(X, for_optimization=False)
    