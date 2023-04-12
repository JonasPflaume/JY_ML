from asopt.base import ProblemTemplate
from asopt.base import FT
import numpy as np
from numba import jit

class HalfCircle(ProblemTemplate):
    ''' A constrained toy problem: x \in R^2
        f(x) = x - y
                x = 0
        x^2 + y^2 <= 1
        optimal result at: x* = (0,1)
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @staticmethod
    @jit(nopython=True)
    def evaluate(x, returnJ, returnH):

        x0, x1 = x[0,0], x[1,0]
        f = x0 - x1
        eq = x0
        ineq = x0**2 + x1**2 - 1
        phi = np.array([[f],[eq],[ineq]],dtype=np.float64)

        if not returnJ:
            return [phi]
        if returnJ:
            J = np.array([[1.,-1.],[1.,0.],[2.*x0, 2.*x1]],dtype=np.float64)
            
            if returnH:
                H =  np.zeros((2,2),dtype=np.float64)
                return [phi, J, H]
            else:
                return [phi, J]

    def getDimension(self):
        return 2

    def getFeatureTypes(self):
        return [FT.obj, FT.eq, FT.ineq]

    def getInitializationSample(self):
        return np.array([[2.],[2.]]) # Not a interior point !

class Rosenbrock2D(ProblemTemplate):
    ''' An unconstrained problem
        f(x,y) = (a-x)^2 + b(y-x^2)^2
        a = 1, b = 100
        optimal result at: x* = (1,1)
    '''
    def __init__(self):
        super().__init__()
    
    @staticmethod
    @jit(nopython=True)
    def evaluate(x, returnJ, returnH, a=1., b=100.):
        x0, x1 = x[0,0], x[1,0]
        f = (a - x0) ** 2 + b * (x1 - x0 ** 2) ** 2
        phi = np.array([[f]],dtype=np.float64)
        
        if not returnJ:
            return [phi]
        if returnJ:
            J11 = 2 * (x0-a) + 4 * b * x0 * (x0**2 - x1)
            J22 = 2 * b * (x1 - x0**2)
            J = np.array([[J11,J22]],dtype=np.float64)
            
            if returnH:
                H11 = 2 + 12 * b * x0 ** 2 - 4 * b * x1
                H12 = -4 * b * x0
                H21 = H12
                H22 = 2 * b
                H = np.array([[H11,H12],[H21,H22]],dtype=np.float64)
                return [phi, J, H]
            else:
                return [phi, J]

    def getDimension(self):
        return 2

    def getFeatureTypes(self):
        return [FT.obj]

    def getInitializationSample(self):
        return np.array([[0.],[0.]])
    
    def getArgsArr(self):
        return np.array([self.a, self.b])
