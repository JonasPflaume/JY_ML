import casadi as cs
import numpy as np
from scipy.linalg import block_diag
import os
import torch as th

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
    
    def plan(self, x0, solver='ipopt'):
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
            if solver == 'ipopt':
                self.solver = cs.nlpsol('solver','ipopt', prob)
            elif solver == 'qp':
                self.solver = cs.qpsol('solver', 'qpoases', prob)
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
        # Jk = - cs.cos(x[0]) * self.system.l * self.system.m * self.system.g + 1/2 * self.system.J * x[1] ** 2
        Jk = 100*x[0] ** 2 + x[1] ** 2
        func = cs.Function("state_cost", [x, u], [Jk])
        return func
    
    def set_terminal_cost_expr(self, x):
        # JK = - cs.cos(x[0]) * self.system.l * self.system.m * self.system.g + 1/2 * self.system.J * x[1] ** 2
        JK = 100*x[0] ** 2 + x[1] ** 2
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
    
class MLP_DMD_MPC_Pendulum(MultiShooting_MPC):
    def __init__(self, system, A, B, horizon) -> None:
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        target_dir = os.path.dirname(curr_dir)
        target_dir = os.path.join(target_dir, "jylearn/timeseries/models")
        system.nx = A.shape[0]
        system.nu = B.shape[1]
        super().__init__(system, horizon)
        
        x, u = cs.SX.sym('x', A.shape[1]), cs.SX.sym('u', B.shape[1])
        
        x_next = A @ x + B @ u
        self.x_next = cs.Function("sys", [x,u], [x_next])

        self.state_cost_func = self.set_stage_cost_func(x, u)
        self.terminal_cost_func = self.set_terminal_cost_expr(x)
        
    def set_stage_cost_func(self, x, u):
        Jk = - cs.cos(x[0]) * self.system.l * self.system.m * self.system.g + 1/2 * self.system.J * x[1] ** 2 + 0.01*u**2
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
        lb_x = np.concatenate([lb_x, -np.ones([self.system.nx-2, 1])*1000], axis=0)
        ub_x = np.concatenate([ub_x, np.ones([self.system.nx-2, 1])*1000], axis=0)

        # delta_u changing rate per step constraints
        lb_u = -2*np.ones((1, 1))
        ub_u = 2*np.ones((1, 1))
        
        return lb_x, ub_x, lb_u, ub_u
    
################################################################

class Condensed_MPC_MLP_EDMD:
    def __init__(self, A, B, horizon) -> None:
        '''
        '''
        self.horizon = horizon
        H, G, L, M, c = self.initialize_qp_problem(A, B)
        self.solver, self.ubg = self.initialize_qp_solver(H, G, L, M, c)
        
    def initialize_qp_solver(self, H, G, L, M, c):
        U = cs.SX.sym('U', self.horizon)
        z0 = cs.SX.sym('z0', M.shape[1])
        f = cs.simplify( U.T @ H @ U + z0.T @ G @ U )
        g = cs.simplify( L @ U + M @ z0 - c )
        qp = {'x':cs.vertcat(U), 'f':f, 'g':g, 'p':z0}
        S = cs.qpsol('S', 'qpoases', qp)
        
        ubg = [0. for i in range(g.shape[1])]
        return S, ubg
        
    def initialize_qp_problem(self, A, B):
        nx, nu = B.shape
        
        Qi = np.zeros([nx, nx])
        Qi[0,0] = 100
        Qi[1,1] = 1
        
        r_factor = 0.
        Ri = np.eye(nu) * r_factor
        
        Ei = np.zeros([6,nx])
        Ei[0,0] = 1.
        Ei[1,1] = 1.
        Ei[2,0] = -1.
        Ei[3,1] = -1.
        
        ENp = np.zeros([4,nx])
        ENp[0,0] = 1.
        ENp[1,1] = 1.
        ENp[2,0] = -1.
        ENp[3,1] = -1.
        
        Fi = np.zeros([6,1])
        Fi[4,0] = 1.
        Fi[5,0] = -1.
        
        # FNp = np.zeros([4,1])
        
        bi = np.array([200., 8, 200, 8, 2, 2]).reshape(-1,1) # constraints (x,u)
        bNp = np.array([200., 8, 200, 8]).reshape(-1,1)
        
        A_stack = [np.linalg.matrix_power(A, i) for i in range(self.horizon+1)]
        A_stack = np.concatenate(A_stack, axis=0)
        
        B_stack = np.zeros([self.horizon*A.shape[0], self.horizon*B.shape[1]])
        for i in range(self.horizon):
            B_stack += np.kron( np.eye(self.horizon, k=-i), np.linalg.matrix_power(A, i)@B )
        B_stack = np.concatenate([np.zeros([B.shape[0], B.shape[1]*self.horizon]), B_stack], axis=0)
            
        F = np.kron(np.eye(self.horizon), Fi)
        F = np.concatenate([F, np.zeros([Fi.shape[0]-2, Fi.shape[1]*self.horizon])], axis=0)
        
        Q = np.kron( np.eye(self.horizon+1), Qi )
        R = np.kron( np.eye(self.horizon), Ri )
        E = block_diag(*([Ei for i in range(self.horizon)] + [ENp]))
        
        # build the parameters
        H = R + B_stack.T @ Q @ B_stack
        G = 2 * A_stack.T @ Q @ B_stack
        L = F + E @ B_stack
        M = E @ A_stack
        c = np.concatenate([bi for i in range(self.horizon)]+[bNp], axis=0)
        return H, G, L, M, c
    
    def plan(self, x0):
        res = self.solver(x0=np.ones(self.horizon)*0.1, ubg=self.ubg, p=x0)
        Ures = res['x'].full()
        return Ures
        
from system import Pendulum
import matplotlib.pyplot as plt
from aslearn.parametric.mlp import MLP

curr_dir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.dirname(curr_dir)
target_dir = os.path.join(target_dir, "jylearn/timeseries/models")

hyper = {"layer":5, "nodes":[2,6,18,36,60], "actfunc":["ReLU", "ReLU", "ReLU", None]}
lifting = MLP(hyper).double()
lifting.load_model(target_dir + "/Epoch_1000_VLoss_5.57_net.pth")
A, B = th.load(target_dir + "/Epoch_1000_VLoss_5.57_A.pth").numpy(), th.load(target_dir + "/Epoch_1000_VLoss_5.57_B.pth").numpy()

rewards = []
horizon = 12
sys = Pendulum()
controller = MLP_DMD_MPC_Pendulum(sys, A, B, horizon)

# for _ in range(50):
#     rewards.append([])
done = False
sys = Pendulum()
x0 = sys.x
x0_l = lifting(th.from_numpy(x0))
x0 = np.concatenate([x0, x0_l.detach().numpy()])
x_his = []
u_his = []
for _ in range(100):
    X, U = controller.plan(x0)
    u_his.append(U[0:1,:])
    x_his.append(x0[:2].reshape(1,-1))
    curr_x, reward, done  = sys.step(U[0,:], render=True)
    # rewards[-1].append(reward)

    x0 = curr_x
    x0_l = lifting(th.from_numpy(x0))
    x0 = np.concatenate([x0, x0_l.detach().numpy()])
    
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "Helvetica",
#     "font.size": 15
# })
# x_his, u_his = np.concatenate(x_his), np.concatenate(u_his)
# plt.subplot(211)
# plt.plot(x_his[:,0], label=r'$\theta$')
# plt.plot(x_his[:,1], label=r'$\dot{\theta}$')
# # plt.xlabel("time step")
# plt.ylabel("States")
# plt.grid()
# plt.legend()
# plt.subplot(212)
# plt.plot(u_his, label=r'$u$')
# plt.grid()
# plt.xlabel("time step")
# plt.ylabel("Control N/m")
# plt.legend()
# plt.tight_layout()
# plt.savefig("/home/jiayun/Desktop/fig.jpg", dpi=150)
# plt.show()
# epi_rewards = np.array([sum(elem) for elem in rewards])
# mean_reward = epi_rewards.mean()
# var_reward = epi_rewards.var()
# plt.boxplot(epi_rewards)
# plt.show()
# print("Mean episode reward: %.2f" % mean_reward, "Var episode reward: %.2f" % var_reward)