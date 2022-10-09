from re import U
import numpy as np
import gym
import casadi as cs
from scipy.linalg import expm

class Pendulum:
    ''' Utilize the render of gym. 
        The system function was implemented though casadi 
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

        self.continuous_nls, self.continuous_ls \
            = self.__init_nonlinear_sys()
        
    def __init_nonlinear_sys(self):
        x = cs.SX.sym("x", 2)
        u = cs.SX.sym("u", 1)
        x1dot = x[1]
        x2dot = self.g / self.l * cs.sin(x[0]) + 1 / (self.m * self.l**2) * u
        f = cs.vertcat(x1dot, x2dot)
        f_func = cs.Function("f_func", [x, u], [f])
        nl_sys = {'f': f_func}
        
        A = cs.jacobian(f, x)
        B = cs.jacobian(f, u)
        A_func = cs.Function('A_func', [x, u], [A])
        B_func = cs.Function('B_func', [x, u], [B])
        l_sys = {'A_func':A_func, 'B_func':B_func}
        return nl_sys, l_sys
    
    def get_discrete_sys(self, x0:np.ndarray, u0:np.ndarray) -> dict:
        ''' get the discrete time system use the linearized system at point x
        '''
        A_func = self.continuous_ls['A_func']
        B_func = self.continuous_ls['B_func']
        A = A_func(x0, u0).full()
        B = B_func(x0, u0).full()
        A_d = expm(A*self.dt)
        B_d = (A_d - np.eye(A_d.shape[0])) @ np.linalg.inv(A) @ B
        
        return A_d, B_d
    
    def step(self, ut, render=False):
        ''' step action
        '''
        obs, rewards, dones, info = self.physical_sys.step(ut)
        self.x = self.physical_sys.state
        if render:
            self.physical_sys.render()