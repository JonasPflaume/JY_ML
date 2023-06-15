import numpy as np
from numba import jit
from asopt.base import FT

@jit(nopython=True)
def btls_increase(lr, rho_a_in=1.2, sigmax=1.):
    increased_lr = np.min(np.array([lr * rho_a_in, sigmax]))
    return increased_lr

@jit(nopython=True)
def btls_decrease(lr, curr_x, direction, evaluate, rho_ls=0.01, rho_a_de=0.5):
    while True:
        lhs_x = curr_x + lr * direction
        temp = evaluate(lhs_x, returnJ=False, returnH=False)
        lhs = temp[0]
        temp = evaluate(curr_x, returnJ=True, returnH=False)
        rhs, J = temp[0], temp[1]
        grad = J.T
        rhs += rho_ls * grad.T @ (lr * direction)
        if lr < 1e-16:
            print("--- WARNING: Can't find a proper lr, breaking ---")
            break
        if lhs > rhs:
            lr *= rho_a_de
        else:
            break
    return lr

@jit(nopython=True)
def get_direction(J, H, verbose):
    grad = J.T
    try:
        # non-positive fall-back
        direction = - np.linalg.solve(H, grad)
    except:
        print('--- PULLBACK ---')
        direction = - grad
        
    direction = np.ascontiguousarray(direction)
    if np.any(grad.T @ direction > 0):
        # wolfe-condition
        if verbose:
            print('--- PULLBACK ---')
        direction = -grad

    return direction

@jit(nopython=True)
def uc_main(x0, lr, tolerance, evaluate, verbose):
    i = 0
    while True:
        # 1. get the update direction
        temp = evaluate(x0, returnJ=True, returnH=True) #self.AulaEvaluate(x0, mu, labd, v, kappa)
        phi, J, H = temp[0], temp[1], temp[2]
        
        if verbose:
            print("Step: ", i, " FuncVal: ", phi[0,0])
            print("lr: ", lr)
        direction = get_direction(J, H, verbose)

        # 2. run the BT linear search
        lr = btls_decrease(lr, x0, direction, evaluate)
        
        # 4. break loop
        if np.linalg.norm(lr*direction) < tolerance:
            x1 = x0
            break
        # 3. update x
        x1 = x0 + lr * direction
        
        x0 = x1
        lr = btls_increase(lr)
        i += 1
    return x1
    
def unconstrained_opt_solve(x0, evaluate, problem_FT, tolerance = 1e-6, verbose=False):
    '''
        Function-style programming enables jit.
        we aim to achieve a competitively high run speed.
    '''
    assert len(problem_FT) == 1 and problem_FT[0] == FT.obj, "Please use constrained optimization solver."

    lr = 1.0
    
    opt_res = uc_main(x0, lr, tolerance, evaluate, verbose)
    
    return opt_res

if __name__ == "__main__":
    # comparison with scipy.minimize, 
    # to showcase the superior efficient brought by jit
    
    # With this rather sloppy implementation, we achieved 50 times faster convergence than scipy minimize.
    # roughly estimated, the jit decorater brings 50 times improvement of run speed.
    from toyproblems import Rosenbrock2D
    import time
    problem = Rosenbrock2D()
    res = unconstrained_opt_solve(  problem.getInitializationSample(),
                                    problem.evaluate,
                                    problem.getFeatureTypes(), verbose=False)
    
    s = time.time()
    res = unconstrained_opt_solve(problem.getInitializationSample(),
                                problem.evaluate,
                                problem.getFeatureTypes(), verbose=False)
    e = time.time()
    print("Our solver use: {:.2f} ms".format(1000*(e-s)))
    
    from scipy.optimize import minimize
    def rosen(x, a=1., b=100.):
        x0 = x[0]
        x1 = x[1]
        f = (a - x0) ** 2 + b * (x1 - x0 ** 2) ** 2
        
        J11 = 2 * (x0-a) + 4 * b * x0 * (x0**2 - x1)
        J22 = 2 * b * (x1 - x0**2)
        J = np.array([J11, J22])
        return f, J
    
    def hess(x, a=1., b=100.):
        x0 = x[0]
        x1 = x[1]
        H11 = 2 + 12 * b * x0 ** 2 - 4 * b * x1
        H12 = -4 * b * x0
        H21 = H12
        H22 = 2 * b
        H = np.array([[H11,H12],[H21,H22]])
        return H
        
    x0 = np.zeros([2,])
    s = time.time()
    res = minimize(rosen, x0, method='Newton-CG', jac=True, hess=hess,
               options={'gtol': 1e-6, 'disp': True})
    e = time.time()
    print("Scipy solver use: {:.2f} ms".format(1000*(e-s)))