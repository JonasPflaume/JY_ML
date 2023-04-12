##
#   Function-style features with implementations to obtain their jacobian matrices
#   designed for optimization problems that require the jacobian of features.
##
import numpy as np
from numba import jit

@jit(nopython=True)
def feature_fourier(x):
    pass

@jit(nopython=True)
def fourierJacobian(x):
    pass

@jit(nopython=True)
def feature_polynomial(x):
    pass

@jit(nopython=True)
def polynomialJacobian(x):
    pass

@jit(nopython=True)
def feature_one(x, for_optimization=False):
    if for_optimization:
        nx = x.shape[0]
        assert x.shape[1] == 1, "we use convention x \in (nx,1) for optimization."
        X = np.ones((nx+1, 1), dtype=np.float64)
        X[1:,:] = x
        return X
    else:
        N, nx = x.shape
        X = np.ones((N, nx+1), dtype=np.float64)
        X[:,1:] = x
        return X

@jit(nopython=True)
def oneJacobian(x):
    assert x.shape[1] == 1, "we use convention x \in (nx,1) for optimization."
    nx = x.shape[0]
    J = np.zeros((nx+1, nx), dtype=np.float64)
    J[1:,:] = np.eye(nx, dtype=np.float64)
    return J

if __name__ == "__main__":
    # test 
    X = np.random.randn(4,1)
    print(oneJacobian(X))