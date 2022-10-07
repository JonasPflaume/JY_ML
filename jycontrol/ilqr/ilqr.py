import numpy as np
import casadi as cs

class ILQR:
    ''' iterative LQR: casadi was used to conduct the Linearization
    '''
    def __init__(self, system):
        self.system = system