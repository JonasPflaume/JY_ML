import numpy as np

class DMDc:
    def __init__(self):
        ''' the <extended dynamic mode decomposition with control>
            for signal or time series prediction.
            This implementation follows:
              M. Korda and I. Mezić, “Linear predictors for nonlinear dynamical systems: 
              Koopman operator meets model predictive control”
              
            feature_funcs is a list of feature functions
        '''
        
    def feature_transform(self, X, feature_funcs):
        ''' the featuren transformation of a series of feature function
        '''
        X_f = feature_funcs[0](X)
        for i in range(1,len(feature_funcs)):
            X_f = feature_funcs[i](X_f)
        return X_f
        
    def fit(self, X_data, U_data, feature_funcs):
        ''' X-state and U-control
            X_data - (L, n)
            U_data - (L-1, m)
        '''
        self.feature_funcs = feature_funcs
        if type(X_data) == list:
            X, Y, X_f, Y_f, U = self.__data_from_rollouts(X_data, U_data, feature_funcs)
        else:
            X, Y = X_data[:-1,:], X_data[1:,:] # (L-1, n)
            X_f, Y_f = self.feature_transform(X, feature_funcs), self.feature_transform(Y, feature_funcs) # (L-1, f)
            U = U_data
            
        X, Y, X_f, Y_f, U = X.T, Y.T, X_f.T, Y_f.T, U.T

        G = np.concatenate([X_f, U], axis=0)
        G = G @ G.T
        V = Y_f @ np.concatenate([X_f, U], axis=0).T
        M = V @ np.linalg.pinv(G)
        
        A, B = M[:, :X_f.shape[0]], M[:, X_f.shape[0]:]
        C = np.zeros([X.shape[0], A.shape[0]])
        for i in range(X.shape[0]):
            C[i, i+1] = 1.
        print("The dim of lifting space: ", X_f.shape[0], " The data num: ", X_f.shape[1])
        
        self.A, self.B, self.C = A, B, C
        return A, B, C
    
    def __data_from_rollouts(self, X_l, U_l, feature_funcs):
        X, Y = X_l[0][:-1,:], X_l[0][1:,:]
        X_f, Y_f = self.feature_transform(X, feature_funcs), self.feature_transform(Y, feature_funcs)
        U = U_l[0]
        for k in range(1, len(X_l)):
            Xk, Yk = X_l[k][:-1,:], X_l[k][1:,:]
            X = np.concatenate([X, Xk])
            Y = np.concatenate([Y, Yk])
            Xk_f = self.feature_transform(Xk, feature_funcs)
            Yk_f = self.feature_transform(Yk, feature_funcs)
            X_f = np.concatenate([X_f, Xk_f])
            Y_f = np.concatenate([Y_f, Yk_f])
            U = np.concatenate([U, U_l[k]])
        return X, Y, X_f, Y_f, U
    
    def predict(self, x, u):
        ''' one step prediction
            x - (1, n)
            u - (1, m)
        '''
        if hasattr(self, "feature_transform"):
            x_f = self.feature_transform(x, self.feature_funcs)
        else:
            raise LookupError("You should run fit function first.")
        x_f_p = self.A @ x_f.T + self.B @ u.T
        x_p = self.C @ x_f_p
        
        return x_p
    
    def predict_traj(self, x, U):
        ''' trajectory prediction
            x - (1, n)
            U - (T, m)
        '''
        X = []
        if hasattr(self, "feature_transform"):
            pass
        else:
            raise LookupError("You should run fit function first.")
        for ui in U:
            X.append(x.reshape(1,-1))
            x_f = self.feature_transform(x, self.feature_funcs)
            if np.any(x_f == np.NaN) or np.any(x_f == np.inf):
                raise ValueError("The identified system is not stable!")
            x_f_p = self.A @ x_f.T + self.B @ ui.reshape(-1,1)
            x_p = self.C @ x_f_p
            x = x_p.T
        X.append(x.reshape(1,-1))
        X = np.concatenate(X)
        return X
        
if __name__ == "__main__":
    ''' EDMDc example
    '''
    import matplotlib.pyplot as plt
    from jylearn.timeseries.utils import collect_rollouts
    from jycontrol.system import Pendulum
    from jylearn.feature.bellcurve import BellCurve
    from collections import OrderedDict
    from itertools import product
    import pickle
    
    p = Pendulum()
    
    dis_dim = 20
    x1 = np.linspace(-3.*np.pi, 3.*np.pi, dis_dim)
    x2 = np.linspace(-8, 8, dis_dim)
    X_sub = np.concatenate(list(product(x1, x2)) ).reshape(-1, 2)
    order = dis_dim ** 2
    f1 = BellCurve(order, l=1.)
    
    f1.set(X_sub)
    
    f_l = [f1]
    
    # 500 trajs, each has 200 steps.
    # This is designed to has the same env quiries with RL controller SAC.
    X_l, U_l = collect_rollouts(p, 50, 150)
    # with open('train_data.pkl','wb') as f:
    #     pickle.dump([X_l, U_l], f)
        
    # with open('train_data.pkl','rb') as f:
    #     X_l, U_l = pickle.load(f)
        
    edmdc = DMDc()
    A, B, C = edmdc.fit(X_l, U_l, f_l)
    
    # test
    X_test, U_test = collect_rollouts(p, 9, 150) # let's show 9 prediction results
    plt.figure(figsize=[15,10])
    for i in range(9):
        plt.subplot(int("33{}".format(i+1)))
        traj = edmdc.predict_traj(X_test[i][0].reshape(1,-1), U_test[i])
        plt.plot(X_test[i], '-.r', label='ground truth')
        plt.plot(traj, '-b', label='prediction')
        plt.grid()
        if i > 5:
            plt.xlabel("Time Step")
        if i % 3 == 0:
            plt.ylabel("States")
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.tight_layout()
    plt.show()
    
    # X_test, U_test = collect_rollouts(p, 200, 200)
    
    # with open('test_data.pkl','wb') as f:
    #     pickle.dump([X_test, U_test], f)
        
    # with open('test_data.pkl','rb') as f:
    #     X_test, U_test = pickle.load(f)
        
    # vali_loss = 0.
    # for i in range(200):
    #     traj = edmdc.predict_traj(X_test[i][0].reshape(1,-1), U_test[i])
    #     vali_loss += np.linalg.norm(X_test[i] - traj)
    # print(vali_loss/200)
    
    # np.save("A.npy", A), np.save("B.npy", B), np.save("C.npy", C), np.save("bellcurve.npy", X_sub)