import casadi as cs
import numpy as np

class MultiShooting_MPC:
    def __init__(self, system, horizon) -> None:
        self.system = system
        self.horizon = horizon
        
        self.first_call = True
        
    def stage_cost_func(self):
        raise NotImplementedError("Override it through set_stage_cost_func")

    def terminal_cost_func(self):
        raise NotImplementedError("Override it through set_terminal_cost_expr.")
    
    def set_stage_cost_func(self):
        raise NotImplementedError("Please define the cost function by override this method")
    
    def set_terminal_cost_func(self):
        raise NotImplementedError("Please define the cost function by override this method")
    
    def get_constraints(self):
        raise NotImplementedError("Please define the constraints by override this method")
    
    def plan(self, x0):
        nx, nu = self.system.nx, self.system.nu
        if self.first_call:
            J = 0
            X = cs.SX.sym("X", self.horizon*nx, 1)
            U = cs.SX.sym("U", (self.horizon-1)*nu, 1)
            
            lb_X = []
            ub_X = []
            lb_U = []
            ub_U = []
            
            g = []
            lb_g = []
            ub_g = []
            
            for k in range(self.horizon-1):
                xk = X[k*nx:(k+1)*nx, :]
                xk_next = X[(k+1)*nx:(k+2)*nx, :]
                uk = U[k*nu:(k+1)*nu, :]
                
                J += self.state_cost_func(xk, uk)
                
                xk_next_predict = self.x_next(xk, uk)
                
                g.append(xk_next - xk_next_predict)
                lb_g.append(np.zeros((nx,1)))
                ub_g.append(np.zeros((nx,1)))
                
                lb_x, ub_x, lb_u, ub_u = self.get_constraints(k)
                
                lb_X.append(lb_x)
                ub_X.append(ub_x)
                lb_U.append(lb_u)
                ub_U.append(ub_u)

            x_terminal = X[(self.horizon-1)*nx:self.horizon*nx,:]
            J += self.terminal_cost_func(x_terminal)
            
            lb_x, ub_x, lb_u, ub_u = self.get_constraints(self.horizon-1)
            lb_X.append(lb_x)
            ub_X.append(ub_x)
            
            self.lbx = cs.vertcat(*lb_X, *lb_U)
            self.ubx = cs.vertcat(*ub_X, *ub_U)
            x = cs.vertcat(X, U)
            g = cs.vertcat(*g)
            self.lbg = cs.vertcat(*lb_g)
            self.ubg = cs.vertcat(*ub_g)

            prob = {'f':J, 'x':x, 'g':g}
            self.solver = cs.nlpsol('solver','ipopt', prob)
            self.lbx[:nx]=x0
            self.ubx[:nx]=x0

            res = self.solver(lbx=self.lbx,ubx=self.ubx,lbg=self.lbg, ubg=self.ubg)
            Xres = res['x'][:self.horizon*nx].full().reshape(self.horizon, nx)
            Ures = res['x'][self.horizon*nx:].full().reshape(self.horizon-1, nu)
            
            self.first_call = False
        else:
            self.lbx[:nx]=x0
            self.ubx[:nx]=x0

            res = self.solver(lbx=self.lbx,ubx=self.ubx,lbg=self.lbg,ubg=self.ubg)
            Xres = res['x'][:self.horizon*nx].full().reshape(self.horizon, nx)
            Ures = res['x'][self.horizon*nx:].full().reshape(self.horizon-1, nu)
        return Xres, Ures
    

class DMD_MPC_Pendulum(MultiShooting_MPC):
    def __init__(self, system, horizon) -> None:
        super().__init__(system, horizon)
        A, B, C, feature = self.system.get_edmdc_sys()
        x, u = self.system.x_sym, self.system.u_sym
        
        x_next = C @ (A @ feature(x, casadi=True) + B @ u)
        self.x_next = cs.Function("sys", [x,u], [x_next])
        
        # x_next = self.system.x_next_rk4
        # self.x_next = cs.Function("sys", [x, u], [x_next])
        self.state_cost_func = self.set_stage_cost_func(self.system.x_sym, self.system.u_sym)
        self.terminal_cost_func = self.set_terminal_cost_expr(self.system.x_sym)
        
    def set_stage_cost_func(self, x, u):
        Jk = - cs.cos(x[0]) * self.system.l * self.system.m * self.system.g + 1/2 * self.system.J * x[1] ** 2
        func = cs.Function("state_cost", [x, u], [Jk])
        return func
    
    def set_terminal_cost_expr(self, x):
        JK = - cs.cos(x[0]) * self.system.l * self.system.m * self.system.g + 1/2 * self.system.J * x[1] ** 2
        func = cs.Function("terminal_cost", [x], [JK])
        return func
    
    def get_constraints(self, k):
        ''' time varying bounds for x and u
        '''
        lb_x = np.array([[-20], [-8]])
        ub_x = np.array([[20], [8]])

        # delta_u changing rate per step constraints
        lb_u = -2*np.ones((1, 1))
        ub_u = 2*np.ones((1, 1))
        
        return lb_x, ub_x, lb_u, ub_u
        
class RK4_MPC_Pendulum(MultiShooting_MPC):
    def __init__(self, system, horizon) -> None:
        super().__init__(system, horizon)
        x, u = self.system.x_sym, self.system.u_sym
        x_next = self.system.x_next_rk4
        self.x_next = cs.Function("sys", [x, u], [x_next])
        self.state_cost_func = self.set_stage_cost_func(self.system.x_sym, self.system.u_sym)
        self.terminal_cost_func = self.set_terminal_cost_expr(self.system.x_sym)
        
    def set_stage_cost_func(self, x, u):
        Jk = - cs.cos(x[0]) * self.system.l * self.system.m * self.system.g + 1/2 * self.system.J * x[1] ** 2
        func = cs.Function("state_cost", [x, u], [Jk])
        return func
    
    def set_terminal_cost_expr(self, x):
        JK = - cs.cos(x[0]) * self.system.l * self.system.m * self.system.g + 1/2 * self.system.J * x[1] ** 2
        func = cs.Function("terminal_cost", [x], [JK])
        return func
    
    def get_constraints(self, k):
        ''' time varying bounds for x and u
        '''
        lb_x = np.array([[-20], [-8]])
        ub_x = np.array([[20], [8]])

        # delta_u changing rate per step constraints
        lb_u = -2*np.ones((1, 1))
        ub_u = 2*np.ones((1, 1))
        
        return lb_x, ub_x, lb_u, ub_u
        
from system import Pendulum
import matplotlib.pyplot as plt

rewards = []
for _ in range(50):
    rewards.append([])
    done = False
    
    sys = Pendulum()
    horizon = 30
    controller = DMD_MPC_Pendulum(sys, horizon)
    x0 = sys.x
    while not done:
        X, U = controller.plan(x0)
        curr_x, reward, done  = sys.step(U[0,:], render=False)
        rewards[-1].append(reward)
        # curr_x = sys.step_openloop(x0, U[0,:])
        x0 = curr_x
    
epi_rewards = np.array([sum(elem) for elem in rewards])
mean_reward = epi_rewards.mean()
var_reward = epi_rewards.var()
plt.boxplot(epi_rewards)
plt.show()
print("Mean episode reward: %.2f" % mean_reward, "Var episode reward: %.2f" % var_reward)