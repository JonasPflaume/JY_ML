from asopt.base import FT
from numba import jit
import numpy as np

@jit(nopython=True)
def btls_increase(lr, rho_a_in=1.2, sigmax=1.):
    increased_lr = np.min(np.array([lr * rho_a_in, sigmax]))
    return increased_lr

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
def aula_main(x0, lr, theta_tol, epsi_tol, uc_tolerance, evaluate, obj_idx, eq_idx, ineq_idx, verbose, *args):
    ''' where the augmented lagrangian main loop defined 
    '''
    # initialize the parameters
    rho_mu = 3.0; rho_v = 3.0; mu = 1.0; v = 1.0
    
    # solveUC will handle the empty list
    labd = np.zeros((len(ineq_idx),1),dtype=np.float64)
    kappa = np.zeros((len(eq_idx),1),dtype=np.float64)

    i = 0
    while True:
        if verbose:
            print("--- iteration " , i, " ---")
        # 1. solve the unconstrained optimization
        x1 = aula_uc_main(x0, lr, mu, labd, v, kappa, obj_idx, eq_idx, ineq_idx, evaluate, uc_tolerance, verbose, *args)
        # 2. update the parameters
        phi, J = evaluate(x1, True, False, *args)
        if ineq_idx.size > 0:
            g_x = phi[ineq_idx]
            labd = np.maximum(labd + 2.0 * mu * g_x, np.zeros_like(labd))
        if eq_idx.size > 0:
            h_x = phi[eq_idx]
            kappa += 2.0 * v * h_x
        # 3. * try to update the mu and v (LP:this will increase the complexity in inner loop)
        mu *= rho_mu
        v *= rho_v
        # 4. check the stop criterion by enumeration
        if ineq_idx.size > 0 and not eq_idx.size > 0:
            if np.linalg.norm(x1-x0) < theta_tol and np.all(g_x < epsi_tol):
                break
        elif eq_idx.size > 0 and not ineq_idx.size > 0:
            if np.linalg.norm(x1-x0) < theta_tol and np.all(np.abs(h_x) < epsi_tol):
                break
        elif not eq_idx.size > 0 and not ineq_idx.size > 0:
            if np.linalg.norm(x1-x0) < theta_tol:
                break
        elif eq_idx.size > 0 and ineq_idx.size > 0:
            if np.linalg.norm(x1-x0) < theta_tol and np.all(g_x < epsi_tol) and np.all(np.abs(h_x) < epsi_tol):
                break
        x0 = x1
        i += 1

    return x1, phi[0,0]

@jit(nopython=True)
def btls_decrease(lr, curr_x, mu, labd, v, kappa, obj_idx, eq_idx, ineq_idx, evaluate, direction, *args):
    rho_ls=0.01
    rho_a_de=0.5
    while True:
        lhs_x = curr_x + lr * direction
        temp = aula_evaluate(lhs_x, mu, labd, v, kappa, obj_idx, eq_idx, ineq_idx, evaluate, *args)
        lhs = temp[0]
        temp = aula_evaluate(curr_x, mu, labd, v, kappa, obj_idx, eq_idx, ineq_idx, evaluate, *args)
        rhs, J = temp[0], temp[1]
        grad = J.T
        rhs += rho_ls * grad.T @ (lr * direction)
        if lr < 1e-20:
            print("--- WARNING: Can't find a proper lr, breaking ---")
            break
        if lhs > rhs:
            lr *= rho_a_de
        else:
            break
    return lr

@jit(nopython=True)
def aula_uc_main(x0, lr, mu, labd, v, kappa, obj_idx, eq_idx, ineq_idx, evaluate, uc_tolerance, verbose, *args):
    i = 0
    while True:
        # 1. get the update direction
        temp = aula_evaluate(x0, mu, labd, v, kappa, obj_idx, eq_idx, ineq_idx, evaluate, *args)
        phi, J, H = temp[0], temp[1], temp[2]
        
        f_curr = phi[0,0]
        if verbose:
            print("Step: ", i, " FuncVal: ", f_curr)
            print("lr: ", lr)
        direction = get_direction(J, H, verbose)

        # 2. run the BT linear search
        lr = btls_decrease(lr, x0, mu, labd, v, kappa, obj_idx, eq_idx, ineq_idx, evaluate, direction, *args)
        
        # 4. break loop
        if np.linalg.norm(lr*direction) < uc_tolerance or i >100:
            x1 = x0
            break
        # 3. update x
        x1 = x0 + lr * direction
        
        x0 = x1
        lr = btls_increase(lr)
        i += 1
    return x1

@jit(nopython=True)
def aula_evaluate(x, mu, labd, v, kappa, obj_idx, eq_idx, ineq_idx, evaluate, *args):
        ''' wrap the defined problem to get the aula needed phi, gradient and hessian
            input:
                    x0:     the current x value
                    mu:     the current Aula parameters for square term of inequality constraints
                    labd:   the current Aula parameters for augmented lagrangian multiplyer of ineq
                    v:      the current Aula parameters for square term of equality constraints
                    kappa:  the current Aula parameters for augmented lagrangian multiplyer of eq
            output:
                    phi:    feature vector for aula inner loop
                    J:      Jaconbian for aula inner loop
                    H:      Hessian matrix for aula inner loop
        '''

        phi = np.zeros((1,1),dtype=np.float64)
        J = np.zeros((1, x.shape[0]),dtype=np.float64)
        H = np.zeros((x.shape[0], x.shape[0]),dtype=np.float64)

        temp = evaluate(x, True, True, *args)
        phi0, J0, H_f0 = temp[0], temp[1], temp[2]

        if ineq_idx.size > 0:
            I_labd = np.logical_or(phi0[ineq_idx] >= 0, labd > 0)
            I_labd = I_labd.astype(np.float64)
            I_labd = I_labd.reshape(-1,)
            I_labd = np.diag( I_labd ) # diagnal matrix
            g_x = phi0[ineq_idx] # (len(labd),1)
            grad_g_x = J0[ineq_idx] # (len(labd),nx)
            inq_term = ((mu * I_labd @ g_x + labd).T @ g_x).astype(np.float64)
            phi += inq_term
            J += (2.0 * mu * I_labd @ g_x + labd).T @ grad_g_x
            H += 2.0 * mu * grad_g_x.T @ I_labd @ grad_g_x
        if eq_idx.size > 0:
            h_x = phi0[eq_idx]
            grad_h_x = J0[eq_idx]
            eq_term = (v * h_x + kappa).T @ h_x
            phi += eq_term
            J += (2.0 * v * h_x + kappa).T @ grad_h_x
            H += 2.0 * v * grad_h_x.T @ grad_h_x
        if obj_idx.size > 0:
            grad_f_x = J0[obj_idx]
            obj_term = np.sum(phi0[obj_idx])
            phi += obj_term
            J += grad_f_x
            H += H_f0
            
        assert phi.shape == (1,1)
        assert J.shape[0] == 1
        return [phi, J, H]
    
def constrained_opt_solve(x0, evaluate, problem_FT, *args, theta_tol=1e-6, epsi_tol=1e-6, uc_tolerance=1e-6, verbose=False):
    '''
        Function-style programming enables jit.
        we aim to achieve a competitively high run speed.
        
        :param x0               initial x
        :param args             optional args for evaluate function
        
        :param theta_tol        x tolerance                 defalt = 1e-6   
        :param epsi_tol         constraints tolerance       defalt = 1e-6   
        :param uc_tolerance     unconstrained tolerance     defalt = 1e-6
    '''
    
    lr = 1.
    
    obj_idx = np.where(np.squeeze(problem_FT) == FT.obj)[0]
    eq_idx = np.where(np.squeeze(problem_FT) == FT.eq)[0]
    ineq_idx = np.where(np.squeeze(problem_FT) == FT.ineq)[0]

    res = aula_main(x0, lr, theta_tol, epsi_tol, uc_tolerance, evaluate, obj_idx, eq_idx, ineq_idx, verbose, *args)
    x_opt, f_opt = res[0], res[1]
    return x_opt, f_opt

if __name__ == "__main__":
    from toyproblems import HalfCircle
    # the aula solver spends merely .6 ms to solve the constrained problem.
    # The solver needs to be compiled before execution, therefore it takes longer for the first run.
    # However, this solver is designed for our machine learning problem, 
    # and it will be run repeatedly for a considerable long time,
    # so the compile time is, in most case, negligible.
    problem = HalfCircle()

    from time import time
    # not necessarily use 
    res = constrained_opt_solve(problem.getInitializationSample(),
                                problem.evaluate,
                                problem.getFeatureTypes(), 2., verbose=True)
    s = time()

    x_opt, f_opt = constrained_opt_solve(problem.getInitializationSample(),
                                problem.evaluate,
                                problem.getFeatureTypes(), 2.)
    e = time()
    print(x_opt, f_opt)
    print("Time: {:.2f} ms".format((e-s)*1000))