import numpy as np
import matplotlib.pyplot as plt

class ILQR:
    ''' iterative LQR
    '''
    def __init__(self, system, horizon, Q, R, Qf, target):
        self.system = system
        self.horizon = horizon
        self.Q, self.R, self.Qf = Q, R, Qf
        self.target = target
        self.tolerance = 1e-2
        self.max_iter = 100
        
    def plan(self, x0):
        ''' planning main loop
        '''
        u0 = np.zeros([self.horizon-1, self.system.nu]) # (horizon-1, nu)
        traj, l_sys = self.simulation(u0, x0) # (horizon, nx), (horizon-1,)
        delta_u = np.ones_like(u0)

        counter = 0
        while np.all( np.linalg.norm( delta_u , axis=1) > self.tolerance ) and counter < self.max_iter:
            
            counter += 1
            # print("The {}. iteration ...".format(counter))
            
            delta_u = self.step(traj, l_sys, u0)
            
            u0 += delta_u
            
            traj, l_sys = self.simulation(u0, x0)
            
        return traj, u0
        
    def step(self, traj, l_sys, u0):
        ''' forward backward step
        '''
        Sk_p = self.Qf.copy()
        vk_p = self.Qf @ ( traj[-1].reshape(-1,1) - self.target.reshape(-1,1))
        # build the list in positive order
        S_l = []
        v_l = []
        K_l, Kv_l, Ku_l = [], [], []
        
        # backward iteration
        for k in range(1, self.horizon):
            S_l.insert(0, Sk_p.copy()), v_l.insert(0, vk_p.copy())
            
            # unpack the parameters
            xk, uk = traj[-(k+1)].reshape(-1,1), u0[-k].reshape(-1,1)
            Ak, Bk = l_sys[-k]['A_d'], l_sys[-k]['B_d']
            
            # calculate the gain
            common_term = np.linalg.inv(Bk.T @ Sk_p @ Bk + self.R)
            Kk = common_term @ Bk.T @ Sk_p @ Ak
            Kvk = common_term @ Bk.T
            Kuk = common_term @ self.R
            
            K_l.insert(0, Kk.copy()), Kv_l.insert(0, Kvk.copy()), Ku_l.insert(0, Kuk.copy())
            
            Sk_p = Ak.T @ Sk_p @ (Ak - Bk @ Kk) + self.Q
            vk_p = (Ak - Bk @ Kk).T @ vk_p - Kk.T @ self.R @ uk + self.Q @ xk
        S_l.insert(0, Sk_p.copy()), v_l.insert(0, vk_p.copy())
        
        # forward iteration
        delta_x = np.zeros([2, 1])
        delta_u_l = []
        for k in range(self.horizon-1):
            Kk, Kvk, Kuk = K_l[k], Kv_l[k], Ku_l[k]
            vk_p = v_l[k+1]
            Ak, Bk = l_sys[k]['A_d'], l_sys[k]['B_d']
            
            delta_u = - Kk @ delta_x - Kvk @ vk_p - Kuk @ u0[k].reshape(-1,1) # control
            delta_x = Ak @ delta_x + Bk @ delta_u
            delta_u_l.append(delta_u.reshape(1,-1))
            
        delta_u_l = np.concatenate(delta_u_l)
        
        return delta_u_l
        
    def simulation(self, u, x0):
        ''' simulate the system using current u and,
            return the linearized system list
        '''
        traj = []
        l_sys = []
        for ui in u:
            traj.append(x0.reshape(1, -1))
            A_d, B_d = self.system.get_discrete_sys(x0, ui)
            l_sys_i = {'A_d':A_d, 'B_d':B_d}
            l_sys.append(l_sys_i)
            x0 = self.system.step_openloop(x0, ui)
            
        traj.append(x0.reshape(1, -1))
        traj = np.concatenate(traj)
        return traj, l_sys

if __name__ == "__main__":
    from asctr.system import Pendulum
    from tqdm import tqdm
    
    rewards = []
    for round in tqdm(range(50)):
        print(round)
        sys = Pendulum()
        Qf = np.eye(2)
        Q = np.eye(2) * 0 # no stage cost of states
        R = np.eye(1) * 1e-2
        target = np.zeros(2)
        planner = ILQR(sys, 29, Q, R, Qf, target)
        x0 = sys.x
        traj, u = planner.plan(x0)

        done = False
        rewards.append([])
        while not done:
            curr_x, reward, done  = sys.step(u[0], render=False)
            rewards[-1].append(reward)
            x0 = sys.x
            traj, u = planner.plan(x0)
            
    epi_rewards = np.array([sum(elem) for elem in rewards])
    mean_reward = epi_rewards.mean()
    var_reward = epi_rewards.var()
    plt.boxplot(epi_rewards)
    plt.show()
    print("Mean episode reward: %.2f" % mean_reward, "Var episode reward: %.2f" % var_reward)
    