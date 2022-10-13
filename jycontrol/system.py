import numpy as np
import gym
import casadi as cs
import os

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
        self.J = 1/3 * self.m * self.l ** 2
        self.nx = 2
        self.nu = 1

        self.continuous_nls, self.continuous_ls \
            = self.__init_sys()
        
    def __init_sys(self):
        ''' initialize the pendulum system
        '''
        x = cs.SX.sym("x", 2)
        u = cs.SX.sym("u", 1)
        x1dot = x[1]
        x2dot = 3*self.g / (2*self.l) * cs.sin(x[0]) + 3 / (self.m * self.l**2) * u
        f = cs.vertcat(x1dot, x2dot)
        f_func = cs.Function("f_func", [x, u], [f])
        nl_sys = {'f': f_func}
        self.system_expr = {'xdot':f, 'x':x, 'u':u}
        # init the ode integrator
        ode = {'x': x, 'ode': f, 'p': u}
        opts = {'tf': self.dt}
        self.ode_solver = cs.integrator('F', 'idas', ode, opts)
        
        x_p = Pendulum.integrate_RK4(x, u, f, self.dt, N_steps=1)
        self.x_next_rk4 = x_p
        
        A = cs.jacobian(x_p, x)
        B = cs.jacobian(x_p, u)

        A_func = cs.Function('A_func', [x, u], [A])
        B_func = cs.Function('B_func', [x, u], [B]) # residual states linear system
        l_sys = {'A_func':A_func, 'B_func':B_func}
        return nl_sys, l_sys
    
    @staticmethod
    def integrate_RK4(x_expr, u_expr, xdot_expr, dt, N_steps=1):
        ''' implementation of Runge Kutta 4th order method
        '''
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
    
    @property
    def xdot(self):
        ''' get the casadi expression of xdot
        '''
        return self.system_expr.get('xdot')
    
    @property
    def x_sym(self):
        ''' get the casadi expression of x
        '''
        return self.system_expr.get('x')
    
    @property
    def u_sym(self):
        ''' get the casadi expression of u
        '''
        return self.system_expr.get('u')
    
    @property
    def x_next_rk4(self):
        return self.__x_next_rk4
    
    @x_next_rk4.setter
    def x_next_rk4(self, x):
        self.__x_next_rk4 = x
    
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
        return self.x, rewards, dones
            
    def step_openloop(self, xt, ut):
        ''' conduct the open loop simulation
        '''
        res_integrator = self.ode_solver(x0=xt, p=ut)
        x_next = res_integrator['xf']
        xt_p = x_next.full()
        return xt_p
    
    def reset(self):
        self.physical_sys = gym.make("Pendulum-v1", g=9.81)
        obs = self.physical_sys.reset()
        self.x = self.physical_sys.state # init the state
        
        self.continuous_nls, self.continuous_ls \
            = self.__init_sys()
    
    def get_edmdc_sys(self):
        ''' get a global linear system with
            extended dynamic mode decomposition
        '''
        from jylearn.feature.bellcurve import BellCurve
        path = os.path.dirname(os.path.realpath(__file__))
        A, B, C = np.load(path+"/models/pendulum/A.npy"), np.load(path+"/models/pendulum/B.npy"),\
            np.load(path+"/models/pendulum/C.npy")
        bellcurve_hyperparam = np.load(path+"/models/pendulum/bellcurve.npy")
        
        feature = BellCurve(l=1.)
        feature.set(bellcurve_hyperparam)
        
        return A, B, C, feature
    

if __name__ == "__main__":
    # check the RK4
    p = Pendulum()
    rk4 = p.x_next_rk4
    func = cs.Function("next", [p.x_sym, p.u_sym], [rk4])

    Xres = []
    Xres_rk4 = []
    x0 = p.x
    x0_rk4 = x0.copy()
    for i in range(100):
        x0 = p.step_openloop(x0, 1.)
        Xres.append(x0.reshape(1,-1))
        x0_rk4 = func(x0_rk4, 1.).full()
        Xres_rk4.append(x0_rk4.reshape(1,-1))
    Xres = np.concatenate(Xres)
    Xres_rk4 = np.concatenate(Xres_rk4)
    import matplotlib.pyplot as plt
    plt.plot(Xres, '-r')
    plt.plot(Xres_rk4, '-.b')
    plt.show()
            