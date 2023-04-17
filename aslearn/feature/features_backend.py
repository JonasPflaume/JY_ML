##
#   Function-style features with implementations to obtain their jacobian matrices
#   designed for optimization problems that require the jacobian of features.
##
import numpy as np
from numba import jit
import numba as nb

# @jit(nb.float64[:,:](nb.float64[:,:]), nopython=True)
# def feature_fourier(x):
#     pass

# @jit(nb.float64[:,:](nb.float64[:,:]), nopython=True)
# def fourierJacobian(x):
#     pass

@jit(nb.float64[:,:](nb.float64[:,:], nb.int64), nopython=True)
def feature_polynomial(x, degree):
    N = x.shape[0]
    F_dim = x.shape[1]
    X_container = np.zeros((1, 1), dtype=np.float64) # placeholder
    if degree==1:
        return x
    elif degree > 1:
        X_i = feature_polynomial(x, degree-1)
        m = X_i.shape[1]
        # X_container = np.zeros((N, m), dtype=np.float64)
        X_container = X_i
        
        # omit the repetitive feature by this rule
        index_ruler = np.ones((F_dim,), dtype=np.int64)
        start_ruler = 0
        for _ in range(degree-2):
            start_ruler += np.sum(index_ruler)
            for index_ruler_i in range(F_dim):
                index_ruler[index_ruler_i] = np.sum(index_ruler[index_ruler_i:])
            
        # print(X_i)
        # generate the feature
        for colum_index_x in range(F_dim):
            for colum_index_X_i in range(start_ruler, m):
                
                if colum_index_X_i < start_ruler + np.sum(index_ruler)-np.sum(index_ruler[colum_index_x:]):
                    continue
                
                temp = x[:,colum_index_x:colum_index_x+1]*X_i[:,colum_index_X_i:colum_index_X_i+1]
                X_container = np.append(X_container, temp, axis=1)
        return X_container
    else:
        print("You shouldn't give degree lower than 1, the original x was returned...")
        return X_container

# @jit(nb.float64[:,:](nb.float64[:,:]), nopython=True)
# def polynomialJacobian(x):
#     pass

@jit(nb.float64[:,:](nb.float64[:,:]), nopython=True)
def feature_one(x):
    # if for_optimization:
    #     nx = x.shape[0]
    #     assert x.shape[1] == 1, "we use convention x \in (nx,1) for optimization."
    #     X = np.ones((nx+1, 1), dtype=np.float64)
    #     X[1:,:] = x
    #     return X
    # else:
    N, _ = x.shape
    X = np.ones((N, 1), dtype=np.float64)
    X = np.concatenate((X, x), axis=1)
    return X

@jit(nb.float64[:,:](nb.float64[:,:]), nopython=True)
def oneJacobian(x):
    assert x.shape[1] == 1, "we use convention x \in (nx,1) for optimization."
    nx = x.shape[0]
    J = np.zeros((nx+1, nx), dtype=np.float64)
    J[1:,:] = np.eye(nx, dtype=np.float64)
    return J

if __name__ == "__main__":
    # test
    X = np.random.randn(1, 12)
    from sklearn.preprocessing import PolynomialFeatures
    poly = PolynomialFeatures(degree=5)
    print(feature_polynomial(X, 1))
    import time
    s = time.time()
    # print(poly.fit_transform(X)[:,1:])
    print(feature_polynomial(X, 5))
    e = time.time()
    print(e-s)
    # print(feature_polynomial(X, 5))
