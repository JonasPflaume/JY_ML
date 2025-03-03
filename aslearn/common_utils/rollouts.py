import numpy as np

def collect_rollouts(system, num, traj_len, noise_level=0.1):
    ''' sin wave stimulation
    '''
    u_max = system.max_torque
    u_dim = system.nu
    x_dim = system.nx
    X_l = []
    U_l = []
    
    for _ in range(1, num+1):
        counter = 0
        # if np.random.uniform(0,1) > 0.9:
        #     U = np.zeros((traj_len-1, u_dim))
        # else:
        #     U = np.array([np.random.uniform(-u_max, u_max) for _ in range(traj_len-1)])
        phase = np.random.uniform(0.1, 0.5)
        U = np.zeros((traj_len-1, u_dim))#0.2*np.sin( phase * np.linspace(0,10,traj_len-1) )
        U = np.tile(U.reshape(-1, 1), (u_dim, 1))
        
        X_l.append([])
        system.reset()
        X_l[-1].append(system.x.reshape(1,-1).copy())
        x0 = system.x
        for ui in U:
            counter += 1
            x_curr = system.step_openloop(x0, ui)
            # x_curr += noise_level*np.random.randn(x_dim,)
            X_l[-1].append(x_curr.reshape(1,-1).copy())
            # if np.any( np.abs(x_curr) > np.array([2.5*np.pi, 20])):
            #     break
            x0 = x_curr
        U_l.append(U[:counter,:].copy())
    X_l = [np.concatenate(elem) for elem in X_l]
    
    
    return X_l, U_l