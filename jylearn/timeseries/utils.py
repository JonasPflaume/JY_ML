from re import U
import numpy as np

def collect_rollouts(system, num, traj_len, max_excitation_ratio=None):
    ''' sin wave stimulation
    '''
    u_max = system.max_torque
    u_dim = system.nu
    X_l = []
    U_l = []
    for _ in range(1, num+1):
        if max_excitation_ratio != None:
            U = max_excitation_ratio * u_max * (np.cos(2.5*np.linspace(0., 2*np.pi, traj_len-1)) +\
                np.sin(1.3*np.linspace(0., 2*np.pi, traj_len-1)))
        else:
            U = np.random.uniform(-u_max, u_max, size=[traj_len-1,])
        U = np.tile(U.reshape(-1, 1), (u_dim, 1))
        
        U_l.append(U.copy())
        X_l.append([])
        system.reset()
        X_l[-1].append(system.x.reshape(1,-1).copy())
        for ui in U:
            x_curr, rewards, dones = system.step(ui)
            X_l[-1].append(x_curr.reshape(1,-1).copy())
        
    X_l = [np.concatenate(elem) for elem in X_l]
    
    return X_l, U_l