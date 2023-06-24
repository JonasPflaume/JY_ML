import numpy as np
from utils import RBF, integrate_RK4
from canonical_system import canonical_system, canonical_system_linear
import copy

class FlexDMP:
    ''' <Motivation:    Higher order DMP>
        This DMP was implemented in vectorized form,
        unlike the implemented vanilla DMP
    '''
    def __init__(self, bf_num, demo_len, demo_dt, p=2.5, w_factor=0.5, tau=1., dof=1, linear_decay=True): # p=23, w = 1.3 better for peak violation!
        '''
        bf_num:     The number of Psi activation points
        demo_len:   The length of the trajectory
        demo_dt:    The time interval of demonstration
        p:          The value to determine the alpha_{1,2,3}

        Marks:      The canonical system can pass in a variable has same length as demo_len to adjust the speed of state x
        '''
        if linear_decay:
            self.cs = canonical_system_linear(demo_len, demo_dt, tau=tau)
        else:
            self.cs = canonical_system(demo_len, demo_dt, tau=tau)
        self.bf_num = bf_num
        self.demo_len = demo_len
        self.demo_dt = demo_dt
        self.tau = tau
        self.dof = dof
        # desired activations throughout time
        if type(self.cs) == canonical_system:
            des_c = np.linspace(0, self.cs.runtime, self.bf_num)
            self.c = np.ones(self.bf_num)
            for n in range(self.bf_num):
                # finding x for desired times t, here exp is the solution of the cs system
                self.c[n] = np.exp(-self.cs.ax * des_c[n])

            self.h = w_factor*np.ones(self.bf_num) * self.bf_num ** 2 / self.c / self.cs.ax  # width increase 20 times (care for the local!)
            self.rbf_l = [RBF(center, width) for center, width in zip(self.c, self.h)]
        elif type(self.cs) == canonical_system_linear:
            self.c = np.linspace(1, 0, self.bf_num)
            sigma = np.abs(self.c[1]-self.c[0])/2 * np.sqrt(-1/(2*np.log(0.5))) * np.ones(self.bf_num)
            self.h = 1 / (2*sigma**2)
            self.h *= w_factor
            self.rbf_l = [RBF(center, width) for center, width in zip(self.c, self.h)]
        
        self.timesteps = demo_len
        # calculate the factors in ODE
        self.alpha1, self.alpha2, self.alpha3 = 3*p, p, 1/3*p               # hand tuned the factor p
    
    def get_weights(self, y_demo, dy_demo=None, ddy_demo=None):
        '''
        The input trajectories should have shape (length, dof)
        '''
        assert y_demo.shape[1] == self.dof, "Please input the trajectory has shape (length, dof)"
        self.y10 = copy.deepcopy(y_demo[0]) # (dof)
        self.goal = copy.deepcopy(y_demo[-1]) # (dof)
        # if there is no dy ddy input
        if type(dy_demo) != np.ndarray and self.dof==1:
            dy_demo = np.gradient(y_demo.squeeze()) / self.demo_dt
            dy_demo = dy_demo[:, np.newaxis]
        elif type(dy_demo) != np.ndarray and self.dof>1:
            dy_demo = np.gradient(y_demo)[0] / self.demo_dt
        if type(ddy_demo) != np.ndarray and self.dof==1:
            ddy_demo = np.gradient(dy_demo.squeeze()) / self.demo_dt
            ddy_demo = ddy_demo[:, np.newaxis]
        elif type(ddy_demo) != np.ndarray and self.dof>1:
            ddy_demo = np.gradient(dy_demo)[0] / self.demo_dt
        
        if self.dof==1:
            dddy_demo = np.gradient(ddy_demo.squeeze()) / self.demo_dt
            dddy_demo = dddy_demo[:, np.newaxis]
        elif self.dof>1:
            dddy_demo = np.gradient(ddy_demo)[0] / self.demo_dt

        self.dy10 = copy.deepcopy(dy_demo[0]) # (dof)
        self.ddy10 = copy.deepcopy(ddy_demo[0]) # (dof)
        # f_target has shape (length, dof)
        f_target = self.tau**3 * dddy_demo + self.alpha1*self.alpha2*self.alpha3*(y_demo-self.goal[np.newaxis, :]) + \
                    self.tau * self.alpha1*self.alpha2 * dy_demo + self.tau**2 * self.alpha1 * ddy_demo
        self.weights = np.zeros([self.dof, self.bf_num])
        x = self.cs.trajectory() # shape (length,)
        s = x[:, np.newaxis]# * ((self.goal - self.y10)[np.newaxis, :]) # shape (length, dof)s
        for i in range(self.bf_num):
            Gamma_i = self.rbf_l[i](x)
            w_i = (s * Gamma_i[:, np.newaxis] * f_target).sum(axis=0) / (s**2 * Gamma_i[:, np.newaxis]).sum(axis=0) # shape (dof,)
            # print(w_i)
            self.weights[:,i] = w_i
        self.weights = np.nan_to_num(self.weights)
        self.reset_state()
        return self.weights

    def set_weight(self, weights, Yinit, dYinit, ddYinit, goal):
        ''' the weights should have the shape (dof, bf_num)
            in the same time you have to give initial and goal of traj with shape (dof,)
        '''
        assert type(weights) == np.ndarray and weights.shape == (self.dof, self.bf_num), "Use the right weight format please!"
        assert type(Yinit) == np.ndarray and Yinit.shape == (self.dof,), "Use the right initial format please!"
        assert type(goal) == np.ndarray and goal.shape == (self.dof,), "Use the right goal format please!"
        self.weights = weights
        self.y10 = Yinit
        self.dy10 = dYinit
        self.ddy10 = ddYinit
        self.goal = goal

    def reset_state(self):
        ''' reset the states for integration
        '''
        self.y = copy.deepcopy(self.y10)
        self.dy = copy.deepcopy(self.dy10)
        self.ddy = copy.deepcopy(self.ddy10)
        self.cs.reset_state()

    def step(self, step_num, timesteps, slow_percent):
        ''' integrate the DMP system step by step
        '''
        # forcing term
        
        x = self.cs.step(step_num, timesteps)

        psi = np.array([self.rbf_l[i](x) for i in range(self.bf_num)]) # (bf_num,)

        # remove the last term to tackle the collapse when very near start and end points were given
        front_term = x# * (self.goal - self.y10) # (dof,)
        f = front_term * np.einsum('ij,j->i', self.weights, psi) # (dof, num) x (num) -> (dof,)
        sum_psi = np.sum(psi)
        if np.abs(sum_psi) > 1e-6:
            f /= sum_psi

        tau = (1 + slow_percent/100) * self.tau
        # self.y += (1/self.tau * self.dy) * self.demo_dt
        # self.dy += (1/(self.tau)*self.ddy) * self.demo_dt
        # self.ddy += (1/self.tau * (self.alpha1 * (self.alpha2 * (self.alpha3*(self.goal-self.y) - self.dy) - self.ddy) + f)) * self.demo_dt

        ## The reality is that RK4 is bad than vanilla euler integration
        def dddy(ddy):
            return (1/tau * (self.alpha1 * (self.alpha2 * (self.alpha3*(self.goal-self.y) - self.dy) - ddy) + f))
        self.ddy = integrate_RK4(self.ddy, dddy, self.demo_dt)

        def dy(y):
            return (1/tau * self.dy)
        self.y = integrate_RK4(self.y, dy, self.demo_dt)

        def ddy(dy):
            return (1/tau * self.ddy)
        self.dy = integrate_RK4(self.dy, ddy, self.demo_dt)
        
        return self.y, 1/tau*self.dy, 1/tau**2*self.ddy # all (dof,)

    def trajectory(self, slow_percent=0):
        ''' reconstruct the whole trajectory
            slow_percent:   you can slow down the trajectory, by adjusting this factor
        '''
        self.reset_state()
        timesteps = int( np.ceil( self.timesteps * (slow_percent/100 + 1) ) )

        # set up tracking vectors
        y_track = np.zeros([timesteps, self.dof])
        dy_track = np.zeros([timesteps, self.dof])
        ddy_track = np.zeros([timesteps, self.dof])

        for t in range(timesteps):
            # run and record timestep
            if t == 0:
                y_track[t, :], dy_track[t, :], ddy_track[t, :] = self.y10, self.dy10, self.ddy10
            else:
                y_track[t, :], dy_track[t, :], ddy_track[t, :] = self.step(t, timesteps, slow_percent)

        return y_track, dy_track, ddy_track
    
if __name__ == "__main__":
    # example
    X = np.linspace(0., 6.28, 101)
    Y = np.sin(X) + np.cos(0.5*X) + np.cos(2*X)
    dY = np.cos(X) - 0.5*np.sin(0.5*X) - 2*np.sin(2*X)
    ddY = -np.sin(X) - 0.25*np.cos(0.5*X) - 4*np.cos(2*X)
    Y = Y.reshape(-1,1)
    dY = dY.reshape(-1,1)
    ddY = ddY.reshape(-1,1)
    
    import matplotlib.pyplot as plt
    from scipy.interpolate import CubicSpline
    
    dmp = FlexDMP(25, len(X), 6.28/101)
    dmp.get_weights(y_demo=Y, dy_demo=dY, ddy_demo=ddY)
    Y_, dY_, ddY_ = dmp.trajectory()
    Y_bundle, dY_bundle, ddY_bundle = [], [], []
    for i in range(20):
        dmp.get_weights(y_demo=Y, dy_demo=dY, ddy_demo=ddY)
        dmp.weights += 0.05 * np.max(dmp.weights, axis=1) * np.random.randn(*dmp.weights.shape)
        Y_i, dY_i, ddY_i = dmp.trajectory()
        Y_bundle.append(Y_i)
        dY_bundle.append(dY_i)
        ddY_bundle.append(ddY_i)
        
    spl = CubicSpline(X[::4], Y[::4])
    Y_spl = spl(X)
    dY_spl = spl(X, nu=1)
    ddY_spl = spl(X, nu=2)
    Ys_bundle, dYs_bundle, ddYs_bundle = [], [], []
    for i in range(20):
        target = (0.05 * np.max(Y[::4], axis=0) * np.random.randn(*Y[::4].shape)).reshape(*Y[::4].shape) + Y[::4]
        spl = CubicSpline(X[::4], target)
        Ys_i = spl(X)
        dYs_i = spl(X,nu=1)
        ddYs_i = spl(X,nu=2)
        Ys_bundle.append(Ys_i)
        dYs_bundle.append(dYs_i)
        ddYs_bundle.append(ddYs_i)
        
    def plot_bundle(Y):
        for yi in Y:
            plt.plot(X, yi, '-c', alpha=0.15)
            
    plt.figure(figsize=[10,7])
    plt.subplot(321)
    plt.title("DMP3: 20 RBFs")
    plt.plot(X, Y, '-.r', linewidth=2.0, label="GT")
    plt.plot(X, Y_, '-b', linewidth=2.0, label="Pred")
    plot_bundle(Y_bundle)
    plt.ylabel("joint angle [rad]")
    plt.grid()
    plt.legend()
    plt.subplot(323)
    plt.plot(X, dY, '-.r', linewidth=2.0)
    plt.plot(X, dY_, '-b', linewidth=2.0)
    plot_bundle(dY_bundle)
    plt.ylabel("joint velocity [rad/s]")
    plt.grid()
    plt.subplot(325)
    plt.plot(X, ddY, '-.r', linewidth=2.0)
    plt.plot(X, ddY_, '-b', linewidth=2.0)
    plot_bundle(ddY_bundle)
    plt.xlabel("time [s]")
    plt.ylabel("joint acc [rad/s^2]")
    plt.grid()
    
    ### spline
    plt.subplot(322)
    plt.title("Cubic Spline: 20 via points")
    plt.plot(X, Y, '-.r', linewidth=2.0)
    plt.plot(X, Y_spl, '-b', linewidth=2.0)
    plot_bundle(Ys_bundle)

    plt.grid()
    plt.subplot(324)
    plt.plot(X, dY, '-.r', linewidth=2.0)
    plt.plot(X, dY_spl, '-b', linewidth=2.0)
    plot_bundle(dYs_bundle)

    plt.grid()
    plt.subplot(326)
    plt.plot(X, ddY, '-.r', linewidth=2.0)
    plt.plot(X, ddY_spl, '-b', linewidth=2.0)
    plot_bundle(ddYs_bundle)
    plt.xlabel("time [s]")
    plt.grid()
    
    plt.tight_layout()
    plt.savefig("/home/jiayun/Desktop/comparison.jpg", dpi=150)
    plt.show()