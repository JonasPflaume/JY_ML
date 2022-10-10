import numpy as np
import gym
import casadi as cs
from scipy.linalg import expm

def integrate_RK4(x_expr, u_expr, xdot_expr, dt, N_steps=1):

    h = dt/N_steps

    x_end = x_expr

    xdot_fun = cs.Function('xdot', [x_expr, u_expr], [xdot_expr])

    for _ in range(N_steps):

        k_1 = xdot_fun(x_end, u_expr)
        k_2 = xdot_fun(x_end + 0.5 * h * k_1, u_expr) 
        k_3 = xdot_fun(x_end + 0.5 * h * k_2, u_expr)
        k_4 = xdot_fun(x_end + k_3 * h, u_expr)

        x_end = x_end + (1/6) * (k_1 + 2 * k_2 + 2 * k_3 + k_4) * h
    
    F_expr = x_end

    return F_expr

class Pendulum:
    ''' Class designed for classical control tests
    
        Utilizing the render of gym. 
        The system function was implemented though casadi,
        to exploit its symbolic differentiation
    '''
    def __init__(self) -> None:
        self.physical_sys = gym.make("Pendulum-v1", g=9.81)
        obs = self.physical_sys.reset()
        self.x = self.physical_sys.state # init the state
        
        # parameters
        self.max_speed = 8
        self.max_torque = 2.0
        self.dt = 0.05
        self.g = 9.81
        self.m = 1.0
        self.l = 1.0
        
        self.nx = 2
        self.nu = 1

        self.continuous_nls, self.continuous_ls \
            = self.__init_nonlinear_sys()
        
    def __init_nonlinear_sys(self):
        x = cs.SX.sym("x", 2)
        u = cs.SX.sym("u", 1)
        x1dot = x[1]
        x2dot = 3*self.g / (2*self.l) * cs.sin(x[0]) + 3 / (self.m * self.l**2) * u
        f = cs.vertcat(x1dot, x2dot)
        f_func = cs.Function("f_func", [x, u], [f])
        nl_sys = {'f': f_func}
        
        # init the ode integrator
        ode = {'x': x, 'ode': f, 'p': u}
        opts = {'tf': self.dt}
        self.ode_solver = cs.integrator('F', 'idas', ode, opts)
        
        x_p = integrate_RK4(x, u, f, self.dt, N_steps=3)
        self.RK4_step_func = cs.Function("RK4", [x, u], [x_p])
        
        A = cs.jacobian(x_p, x)
        B = cs.jacobian(x_p, u)

        A_func = cs.Function('A_func', [x, u], [A])
        B_func = cs.Function('B_func', [x, u], [B])
        l_sys = {'A_func':A_func, 'B_func':B_func}
        return nl_sys, l_sys
    
    def get_discrete_sys(self, x0:np.ndarray, u0:np.ndarray):
        ''' get the discrete time system use the linearized system at point x
            WARNING: This is a system of residual value delta_x and delta_u !!!
        '''
        A_func = self.continuous_ls['A_func']
        B_func = self.continuous_ls['B_func']
        Ad = A_func(x0, u0).full()
        Bd = B_func(x0, u0).full()
        return Ad, Bd
    
    def step(self, ut, render=False):
        ''' step action in the physical system
        '''
        obs, rewards, dones, info = self.physical_sys.step(ut)
        self.x = self.physical_sys.state
        if render:
            self.physical_sys.render()
        return rewards, dones
            
    def step_openloop(self, xt, ut):
        ''' conduct the open loop simulation
        '''
        res_integrator = self.ode_solver(x0=xt, p=ut)
        x_next = res_integrator['xf']
        xt_p = x_next.full()
        return xt_p
    
    def step_RK4(self, xt, ut):
        return self.RK4_step_func(xt, ut).full()