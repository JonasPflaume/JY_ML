import torch as th
import numpy as np
from decimal import Decimal
from scipy.optimize import minimize

device = "cuda" if th.cuda.is_available() else "cpu"

class BSEDMD_1:
    ''' Bayesian shrinkage EDMD through mean-field VI
    '''
    def __init__(self,) -> None:
        '''
        '''
        
    def __init_params(self, nx, ny, prior_impact_factor):
        ''' initialize the variational parameters to make the re-training starts warmly.
        '''
        # initialize model parameter
        # q(W)
        self.M_N = 1e-8*th.randn(ny, nx).to(device).double()
        
        self.U_N = th.eye(ny).to(device).double()
        self.V_N = th.eye(nx).to(device).double()
        
        # q(Lambda)
        self.V_1N = th.eye(nx)
        # self.V_1N = (self.V_1N@self.V_1N.T  + th.eye(nx) * 1e-4)
        self.V_1N = self.V_1N.to(device).double()
        self.n_1N = (nx-1)*th.ones(1,).to(device).double()
        if self.n_1N <= 0.:
            self.n_1N = th.ones(1,).to(device).double()
        
        self.V_2N = th.eye(ny)
        # self.V_2N = (self.V_2N@self.V_2N.T  + th.eye(ny) * 1e-4)
        self.V_2N = self.V_2N.to(device).double()
        self.n_2N = (ny-1) * th.ones(1,).to(device).double()
        if self.n_2N <= 0.:
            self.n_2N = th.ones(1,).to(device).double()
        
        # priors
        # self.V_1_inv = th.randn(nx,nx)
        # self.V_1_inv = 1/100*self.V_1_inv@self.V_1_inv.T
        self.V_1_inv = prior_impact_factor*th.eye(nx)
        self.V_1_inv = self.V_1_inv.to(device).double()
        self.n_1 = nx*th.ones(1,).to(device).double()
        
        # self.V_2_inv = th.randn(ny,ny)
        self.V_2_inv = prior_impact_factor*th.eye(ny)
        # self.V_2_inv = 1/4 * self.V_2_inv @ self.V_2_inv.T
        self.V_2_inv = self.V_2_inv.to(device).double()
        self.n_2 = ny*th.ones(1,).to(device).double() # may cause error
    
    def fit(self, X, Y, max_inter_num=5000, tolerance=1e-9, no_opt=False, prior_impact_factor=2):
        self.nx = X.shape[1]
        self.ny = Y.shape[1]
        self.N = X.shape[0]
        assert len(X)==len(Y), "the time series data should have the same length"
        
        self.__init_params(self.nx, self.ny, prior_impact_factor=prior_impact_factor)
        # prepare fix parameters
        YTX = Y.T @ X
        XTX = X.T @ X
        YTY = Y.T @ Y
        
        # start the fit
        stop_flag = False
        step_counter = 0
        while not stop_flag:
            step_counter += 1
            # q(W)
            Expect_lambda = self.n_1N * self.V_1N
            V_N = XTX + Expect_lambda
            V_N_lower = th.linalg.cholesky(V_N)
            V_N = th.cholesky_inverse(V_N_lower)
            Expect_beta = self.n_2N * self.V_2N
            U_N_lower = th.linalg.cholesky(Expect_beta)
            U_N = th.cholesky_inverse(U_N_lower)
            M_N = YTX @ V_N
            # update
            mean_variation = th.abs(self.M_N - M_N).sum() / self.nx*self.ny

            if step_counter % 10 == 0:
                print("step: ", step_counter, " tolerance: {:.2E}".format(Decimal(mean_variation.detach().cpu().item())))
            if mean_variation < tolerance or step_counter > max_inter_num or no_opt==True:
                stop_flag = True
                
            self.M_N = M_N
            self.U_N = U_N
            self.V_N = V_N
            
            # q(Lambda)
            V_1N = self.V_N * th.trace(self.U_N @ Expect_beta) + self.M_N.T @ Expect_beta @ self.M_N + self.V_1_inv
            V_1N_lower = th.linalg.cholesky(V_1N)
            V_1N = th.cholesky_inverse(V_1N_lower)
            n_1N = self.n_1 + self.ny
            self.V_1N = V_1N
            self.n_1N = n_1N
            
            # q(beta)
            Expect_lambda = self.n_1N * self.V_1N
            S_lambda = XTX + Expect_lambda
            
            V_2N = YTY - 2*YTX @ self.M_N.T + self.U_N*th.trace(S_lambda@self.V_N) + self.M_N @ S_lambda @ self.M_N.T + self.V_2_inv
            V_2N_lower = th.linalg.cholesky(V_2N)
            V_2N = th.cholesky_inverse(V_2N_lower)
            n_2N = self.N + self.nx + self.n_2
            
            self.V_2N = V_2N
            self.n_2N = n_2N
            
        Expect_beta = self.n_2N * self.V_2N
        self.Expect_beta_inv = th.linalg.cholesky(Expect_beta)
        self.Expect_beta_inv = th.cholesky_inverse(self.Expect_beta_inv)
        self.I = th.eye(self.ny).unsqueeze(dim=0).to(device).double() # the identify matrix for prediction
        
        return self
    
    def predict(self, x, return_var=False):
        ''' x - augmented input
        '''
        mean = x @ self.M_N.T
        if return_var == True:
            x_expand = x.unsqueeze(dim=2)
            kron_matrix = th.kron(self.I, x_expand)
            var_temp = th.einsum("ij,bjk->bik", th.kron(self.V_N, self.U_N), kron_matrix)
            var = self.Expect_beta_inv + th.einsum("bij,bjk->bik", th.transpose(kron_matrix, dim0=1, dim1=2), var_temp)
            return mean, var
        else:
            return mean
    
    def horizon_predict(self, x, u, feature_func_list, sample_num=500):
        ''' sampling approach, non-differentiable
            inputs are numpy arrays!
            x - initial state (1,nx), it's not the augmented input as in predict!
            u - control horizon
            feature_func - callable feature transformation
        '''
        def feature_func(x):
            temp = x
            for func in feature_func_list:
                temp = func(temp)
            return temp
        
        with th.no_grad():
            horizon_len = len(u)
            x_dim = x.shape[1]
            u_dim = u.shape[1]
            trajs = np.zeros([sample_num, horizon_len+1, x_dim])
            trajs[:,0,:] = x.squeeze(axis=0)
            
            # the first step
            u_t = u[0:1] # (1,nu)
            x_aug = np.concatenate([x, u_t, feature_func(x)], axis=1)
            x_aug_dim = x_aug.shape[1]
            x_aug = th.from_numpy(x_aug).double().to(device)
            
            x_trans, x_var = self.predict(x_aug, return_var=True)
            
            prior = th.distributions.multivariate_normal.MultivariateNormal(loc=x_trans, covariance_matrix=x_var)
            sampling = prior.sample(sample_shape=th.Size([sample_num])).squeeze(dim=1)
            trajs[:,1,:] = sampling.detach().cpu().numpy()[:,:x_dim]
            
            for i in range(horizon_len-1):
                # modify the control
                u_t = th.from_numpy(u[i+1:i+2]).double().to(device) # (1,nu)
                sampling[:,x_dim:x_dim+u_dim] = u_t
                # transition
                x_trans, x_var = self.predict(sampling, return_var=True)
                    
                # sample the next
                prior = th.distributions.multivariate_normal.MultivariateNormal(loc=x_trans, covariance_matrix=x_var)
                sampling = prior.sample(sample_shape=th.Size([1])).squeeze(dim=0)
                
                trajs[:,2+i,:] = sampling.detach().cpu().numpy()[:,:x_dim]

            return trajs
        
    def horizon_predict_no_prop(self, x, u, feature_func_list):
        ''' no probability propagation
        '''
        horizon_len = len(u)
        x_dim = x.shape[1]
        u_dim = u.shape[1]
        
        def feature_func(x):
            temp = x
            for func in feature_func_list:
                temp = func(temp)
            return temp
        
        u_t = u[0:1]
        trajs = [x]
        vars = [np.zeros([1,x_dim])]
        x_aug = np.concatenate([x, u_t, feature_func(x)], axis=1)
        x_aug = th.from_numpy(x_aug).double().to(device)
        x_trans, x_var = self.predict(x_aug, return_var=True)
        trajs.append(x_trans.detach().cpu().numpy()[:,:x_dim])
        var = th.diag(x_var.squeeze(dim=0))[:x_dim].detach().cpu().numpy().reshape(1,-1)
        vars.append(var)
        for i in range(horizon_len-1):
            u_t = u[i+1:i+2]
            # x = x_trans[:,:x_dim].detach().cpu().numpy()
            # x_aug = np.concatenate([x, u_t, feature_func(x)], axis=1)
            # x_aug = th.from_numpy(x_aug).double().to(device)
            x_trans[:,x_dim:x_dim+u_dim] = th.from_numpy(u_t).double().to(device)
            x_trans, x_var = self.predict(x_trans, return_var=True)

            trajs.append(x_trans.detach().cpu().numpy()[:,:x_dim])
            var = th.diag(x_var.squeeze(dim=0))[:x_dim].detach().cpu().numpy().reshape(1,-1)
            vars.append(var)
        
        trajs = np.concatenate(trajs)
        vars = np.concatenate(vars)
        return trajs, vars    

from aslearn.parametric.vrvm import VRVM

class BSEDMD_2(VRVM):
    ''' output uncorrelated
    '''
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__(input_dim, output_dim)
        
    def horizon_predict_no_prop(self, x, u, feature_func_list):
        ''' no probability propagation
        '''
        horizon_len = len(u)
        x_dim = x.shape[1]
        u_dim = u.shape[1]
        
        def feature_func(x):
            temp = x
            for func in feature_func_list:
                temp = func(temp)
            return temp
        
        u_t = u[0:1]
        trajs = [x]
        vars = [np.zeros([1,x_dim])]
        x_aug = np.concatenate([x, u_t, feature_func(x)], axis=1)
        x_aug = th.from_numpy(x_aug).double().to(device)
        x_trans, x_var = self.predict(x_aug, return_var=True)
        trajs.append(x_trans.detach().cpu().numpy()[:,:x_dim])
        var = x_var[:,:x_dim].detach().cpu().numpy()
        vars.append(var)
        for i in range(horizon_len-1):
            u_t = u[i+1:i+2]
            # x = x_trans[:, :x_dim].detach().cpu().numpy()
            x_trans[:,x_dim:x_dim+u_dim] = th.from_numpy(u_t).double().to(device)
            # x_aug = np.concatenate([x, u_t, feature_func(x)], axis=1)
            # x_aug = th.from_numpy(x_aug).double().to(device)
            x_trans, x_var = self.predict(x_trans, return_var=True)
            
            trajs.append(x_trans.detach().cpu().numpy()[:,:x_dim])
            var = x_var[:,:x_dim].detach().cpu().numpy()
            vars.append(var)
        
        trajs = np.concatenate(trajs)
        vars = np.concatenate(vars)
        return trajs, vars
    
    
if __name__ == "__main__":

    from aslearn.feature.global_features import FourierFT, PolynomialFT
    from asctr.system import Pendulum
    from aslearn.common_utils.rollouts import collect_rollouts
    
    # P = Pendulum()
    # testNum = 8
    # traj_num = 28
    # traj_len = 60
    # X_l, U_l = collect_rollouts(P, num=traj_num, traj_len=traj_len)
    # X = []
    # Y = []
    # U = []
    # for i in range(traj_num - testNum):
    #     Xi = X_l[i]
    #     Ui = U_l[i]
    #     X.append(Xi[:-1])
    #     Y.append(Xi[1:])
    #     U.append(Ui)
    # X = np.concatenate(X)
    # Y = np.concatenate(Y)
    # U = np.concatenate(U)
    # X = X[:-1]
    # Y = Y[:-1]
    # Xtest = X_l[traj_num-testNum:]
    # Utest = U_l[traj_num-testNum:]
    # np.save('X.npy', X)
    # np.save('Y.npy', Y)
    # np.save('U.npy', U)
    # # np.save('X_test.npy', Xtest)
    # # np.save('U_test.npy', Utest)

    # X = np.load('X.npy')
    # # X += np.random.randn(*X.shape)*0.0001
    # Y = np.load('Y.npy')
    # # Y += np.random.randn(*Y.shape)*0.0001
    # U = np.load('U.npy')
    # # Xtest = np.load('X_test.npy')
    # # Utest = np.load('U_test.npy')
    
    # feature1 = FourierFT(degree=[1])
    # feature2 = PolynomialFT(degree=3)
    # Xt = np.concatenate([X, U[:-1], feature2(feature1(X))], axis=1)
    # Yt = np.concatenate([Y, U[1:], feature2(feature1(Y))], axis=1)
    
    # X = th.from_numpy(Xt).double().to(device)
    # Y = th.from_numpy(Yt).double().to(device)
    # print("fea:", X.shape[1])
    # in_dim = X.shape[1]
    # out_dim = in_dim
    # predictor1 = BSEDMD_1().fit(X, Y, max_inter_num=5000) # in_dim, out_dim
    # predictor2 = BSEDMD_2(in_dim, out_dim).fit(X, Y, max_inter_num=1000, tolerance=1e-8) # in_dim, out_dim
    
    # err1 = 0.
    # err2 = 0.
    # for i in range(testNum):
    #     print("test, ", i )
    #     x0 = Xtest[i][0:1]
    #     feature_list = [feature1, feature2]
        
    #     trajs1, vars1 = predictor1.horizon_predict_no_prop(x0, Utest[i], feature_list)
    #     trajs2, vars2 = predictor2.horizon_predict_no_prop(x0, Utest[i], feature_list)

    #     vars1 /= 2 * np.max(vars1)
    #     # vars2 /= 2 * np.max(vars2)
    #     # mean = mean.detach().cpu().numpy()
    #     # var = var.detach().cpu().numpy()
        
    #     err1 += np.linalg.norm(trajs1 - Xtest[i], axis=1).mean()
    #     err2 += np.linalg.norm(trajs2 - Xtest[i], axis=1).mean()
        
    # print(err1/testNum, err2/testNum)
    
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=[10,5])
    # plt.subplot(121)
    # plt.plot(trajs1[:,0], trajs1[:,1], '-b', label="CorSR")
    # # plt.plot(trajs1[:,0], trajs1[:,1]+vars1[:,1], '-.b', alpha=0.4)
    # for i in range(len(trajs1[:,0])):
    #     plt.plot([trajs1[i,0]+vars1[i,0], trajs1[i,0]-vars1[i,0]], [trajs1[i,1], trajs1[i,1]], color='b', alpha=0.4)
    #     plt.plot([trajs1[i,0], trajs1[i,0]], [trajs1[i,1]+vars1[i,1], trajs1[i,1]-vars1[i,1]], color='b', alpha=0.4)
    # # plt.plot(trajs1[:,0]+vars1[:,0], trajs1[:,1]+vars1[:,1],'-.b', alpha=0.4)
    # plt.plot(Xtest[testNum-1][:,0], Xtest[testNum-1][:,1], '-.c', label="GT")
    # plt.xlabel("angle")
    # plt.ylabel("velocity")
    # plt.legend()
    
    # plt.subplot(122)
    # plt.plot(trajs2[:,0], trajs2[:,1], '-b', label="IndSR")
    # for i in range(len(trajs1[:,0])):
    #     plt.plot([trajs2[i,0]+vars2[i,0], trajs2[i,0]-vars2[i,0]], [trajs2[i,1], trajs2[i,1]], color='b', alpha=0.4)
    #     plt.plot([trajs2[i,0], trajs2[i,0]], [trajs2[i,1]+vars2[i,1], trajs2[i,1]-vars2[i,1]], color='b', alpha=0.4)
    # plt.plot(Xtest[testNum-1][:,0], Xtest[testNum-1][:,1], '-.c', label="GT")
    # plt.xlabel("angle")
    # plt.ylabel("velocity")
    # plt.tight_layout()
    # plt.legend()
    # plt.savefig("res_dmd.svg")
    # plt.show()
    
    # prepare the data
    from aslearn.feature.global_features import PolynomialFT, FourierFT
    import matplotlib.pyplot as plt
    X = np.linspace(-15,15,200)[:,np.newaxis]
    sw1 = 0.5*np.sin(1.5*np.pi*0.8*X) - 1.2*np.cos(1.5*np.pi*0.4*X)
    sw2 = 0.5*np.cos(1.5*np.pi*0.8*X) + 1.5*np.sin(1.5*np.pi*0.4*X)
    sw3 = np.sin(2*np.pi*0.4*X)
    Y = np.concatenate([sw1, sw2, sw3], axis=1) + np.random.randn(200,3) * 0.3

    poly = PolynomialFT(degree=2)
    # sqw = SquareWaveFT(frequencies=np.linspace(0.1,10,100))
    fri = FourierFT(degree=np.linspace(1,50,10))

    X_max = np.max(X, axis=0)
    ## normalizing the input is crucial for this framework!
    X = X / X_max
    X_f = poly(fri(X))
    print("Feauture dim: ", X_f.shape[1])
    X_t, Y_t = th.from_numpy(X_f).to(device).double(), th.from_numpy(Y).to(device).double()

    ## fit the model
    blr = BSEDMD_1().fit(X_t, Y_t, no_opt=False)

    ## make predictions
    pred, var = blr.predict(X_t, return_var=True)
    pred, var = pred.detach().cpu().numpy(), var.detach().cpu().numpy()

    print("BLR: {:.2f}".format(np.linalg.norm(pred-Y, axis=1).sum()))

    X = X * X_max
    plt.figure(figsize=[13,4])
    plt.subplot(131)
    plt.plot(X[:,0], Y[:,0], 'r.')
    plt.plot(X[:,0], pred[:,0], 'b-')
    plt.fill_between(X[:,0], pred[:,0]-var[:,0,0], pred[:,0]+var[:,0,0], color='b', alpha=0.4)

    plt.subplot(132)
    plt.plot(X[:,0], Y[:,1], 'r.')
    plt.plot(X[:,0], pred[:,1], 'b-')
    plt.fill_between(X[:,0], pred[:,1]-var[:,1,1], pred[:,1]+var[:,1,1], color='b', alpha=0.4)

    plt.subplot(133)
    plt.plot(X[:,0], Y[:,2], 'r.', label="data")
    plt.plot(X[:,0], pred[:,2], 'b-', label="vRVM")
    plt.fill_between(X[:,0], pred[:,2]-var[:,2,2], pred[:,2]+var[:,2,2], color='b', alpha=0.4)


    plt.legend()
    plt.tight_layout()
    # plt.savefig('res.svg')
    plt.show()