import numpy as np

### not working, theoretical flaw ###
class FiniteLQR:
    ''' A sloppy implementation of finite horizon LQR controller
    '''
    def __init__(self, A, B, C, Q, Qf, R, simNum):
        ''' A, B, C - linear discrete time system
            Q - Stage state penalty matrix
            Qf- terminal state penalty matrix
            R - control cost matrix
            simNum - horizon
        '''
        self.A = A
        self.B = B
        self.C = C
        self.Q = Q
        self.Qf = Qf
        self.R = R
        self.nx = self.A.shape[1]
        self.nu = self.B.shape[1]
        self.nxx = self.C.shape[0]
        self.N = simNum

        self.__LQR_iteration()

    def __LQR_iteration(self):
        N = self.N
        A = self.A
        B = self.B
        # C = self.C
        Q = self.Q
        R = self.R
        Qf = self.Qf
        # Create a list of N + 1 elements
        P = [None] * (N + 1)
        # LQR via Dynamic Programming
        P[N] = Qf
        # Backward iteration
        for i in range(N, 0, -1):
            # Discrete-time Algebraic Riccati equation to calculate the optimal 
            # state cost matrix
            P[i-1] = Q + A.T @ P[i] @ A - (A.T @ P[i] @ B) @ np.linalg.pinv(
                R + B.T @ P[i] @ B) @ (B.T @ P[i] @ A)      
    
        # Create a list of N elements
        self.K = [None] * N
    
        # Forward iteration
        for i in range(N):
            # Calculate the optimal feedback gain K
            self.K[i] = -np.linalg.pinv(R + B.T @ P[i+1] @ B) @ B.T @ P[i+1] @ A
    

    def __call__(self, x, target, step):
        ''' xref for function argument
        '''
        return self.K[step] @ (x - target).reshape(-1, 1)
    
    
if __name__ == "__main__":
    from asctr.system import Pendulum
    import time
    import os
    from aslearn.feature.bellcurve import BellCurve
    
    for _ in range(50):
        sys = Pendulum()
        
        A, B, C, f = sys.get_edmdc_sys()
        
        Qf = np.zeros([A.shape[0], A.shape[0]])
        Qf[1,1] = 1.
        Qf[2,2] = 1.
        Q = Qf
        
        R = np.eye(1) * 1
        horizon = 150
        target = np.zeros(2)
        
        controller = FiniteLQR(A, B, C, Q, Qf, R, horizon)
        x0 = sys.x
        x0_f = f(x0.reshape(1,-1)).squeeze()
        target_f = f(target.reshape(1,-1)).squeeze()
        
        for step in range(horizon):
            u = controller(x0_f, target_f, step)
            curr_x, reward, done  = sys.step(u, render=True)
            x0 = sys.x
            x0_f = f(x0.reshape(1,-1)).squeeze()
        