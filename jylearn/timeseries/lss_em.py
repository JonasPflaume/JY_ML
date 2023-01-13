import torch as th

class LSS(object):
    ''' learning linear state space model by em algorithm
        Implementation by pytorch;
        Two-way update through:
            1. exact M step,
            2. continuous optimization of joint likelihood.
    '''
    def __init__(self, exact_learning:bool=True):
        ''' 
        :exact_learning - if ture, the M step will be conducted though closed form update.
        
        '''
        self.exact_learning = exact_learning
    
    def fit(self, data):
        pass
    
    def predict(self, x):
        pass
    
    def traj_predict(self, x):
        pass