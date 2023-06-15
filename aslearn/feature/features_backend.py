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
def feature_poly_expand(x, degree):
    N = x.shape[0]
    F_dim = x.shape[1]
    X_container = np.zeros((1, 1), dtype=np.float64) # placeholder
    if degree==1:
        return x
    elif degree > 1:
        X_i = feature_poly_expand(x, degree-1)
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

        # generate the feature
        for colum_index_x in range(F_dim):
            for colum_index_X_i in range(start_ruler, m):
                
                if colum_index_X_i < start_ruler + \
                    np.sum(index_ruler)-np.sum(index_ruler[colum_index_x:]):
                    continue
                
                temp = x[:,colum_index_x:colum_index_x+1]*X_i[:,colum_index_X_i:colum_index_X_i+1]
                X_container = np.append(X_container, temp, axis=1)
        return X_container
    else:
        print("You shouldn't give degree lower than 1, the original x was returned...")
        return X_container

@jit(nb.float64[:,:](nb.float64[:,:], nb.int64), nopython=True)
def feature_polynomial(x, degree):
    ''' add the leading one
    '''
    res = feature_poly_expand(x, degree)
    N = x.shape[0]
    ones_prefix = np.ones((N, 1), dtype=np.float64)
    res = np.concatenate((ones_prefix, res), axis=1)
    return res

@jit(nb.float64[:,:](nb.float64[:,:], nb.int64), nopython=True)
def polynomial_jacobian(X, degree):
    ''' two point estimation of polynomial jacobian
    '''
    delta = 1e-5
    dim_x = X.shape[1]
    temp = feature_polynomial(X, degree)
    dim_fx = temp.shape[1]
    
    J = np.zeros((dim_fx, dim_x), dtype=np.float64)
    res = feature_polynomial(X, degree)
    for i in range(dim_x):
        delta_i = np.zeros_like(X)
        delta_i[0,i] = delta
        res_delta = feature_polynomial(X+delta_i, degree)
        jac_i = ((res_delta-res)/delta).reshape(-1,)
        J[:,i] = jac_i
    return J

# change to rbf feature! and implement the conventional rvm
@jit(nopython=True)
def polynomialAL_obj(x, degree, SN):

    ny = SN.shape[0]
    x = x.reshape(1,-1)
    phi_x = feature_polynomial(x, degree)

    f = np.array([[0.]], dtype=np.float64)
    phi_x = np.ascontiguousarray(phi_x)
    
    for i in range(ny):
        f += -0.5 * phi_x @ SN[i] @ phi_x.T
    f = f[0,0]
    return f

@jit(nopython=True)
def polynomialAL_jac(x, degree, SN):
    x = x.reshape(1,-1)
    ny = SN.shape[0]
    phi_x = feature_polynomial(x, degree)
    poly_J = polynomial_jacobian(x, degree)
    poly_J = np.ascontiguousarray(poly_J)
    J_f = np.zeros((1, x.shape[1]), dtype=np.float64)
    for i in range(ny):
        J_f += - phi_x @ SN[i] @ poly_J
    
    J_f = J_f.reshape(-1,)
    return J_f


