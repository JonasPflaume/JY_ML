from enum import Enum

class ProblemTemplate():
    """
    Non Linear program

    min_x    f(x)
    s.t.     phi_eq(x) = 0
             phi_ineq(x) <= 0

    where: 
    x is a continous variable, in vector space R^n
    f is a scalar function
    phi_eq is a vector of equality constraints
    phi_ineq is a vector of inequality constraints
    """

    def __init__(self):
        '''
        '''

    def evaluate(self, x, returnJ, returnH):
        """
        query the NLP at a point x; returns the phi or (phi,J), which is
        the feature vector and its Jacobian; 
        features define cost terms, equalities, and 
        inequalities, their types are depending on 'getFeatureTypes'

        Parameters
        ------
        x: array, (nx,1)

        Returns
        ------
        phi: array (1+eq+ineq,1)
        
        if return J:
        J: array (1+eq+ineq,nx).  J[i,j] is derivative of feature i w.r.t variable j
        
        if return H:
        H: array (nx, nx)

        This design is due to the pervasive dependence between phi, J and H, usually if you want to
        evaluate J, phi has to be known; if H, then J has to be known.
        """
        raise NotImplementedError()

    def getDimension(self):
        """
        return the dimensionality of x

        Returns
        -----
        output: integer

        """
        raise NotImplementedError()

    def getFeatureTypes(self):
        """
        returns
        -----
        output: list of feature Types

        """
        raise NotImplementedError()

    def getInitializationSample(self):
        """
        returns a sample (e.g. uniform within bounds) to initialize an optimization

        Returns
        -----
        x:  (nx,1)

        """
        raise NotImplementedError()
    
    
class FT:
    ''' enum class denoting type of each feature
    '''
    obj     = 1
    eq      = 2
    ineq    = 3