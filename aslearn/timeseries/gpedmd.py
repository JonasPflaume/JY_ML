from aslearn.kernel.kernels import Kernel
from aslearn.kernel.kernels import RBF, White
from aslearn.nonparametric.gpr import ExactGPR
from aslearn.common_utils.check import RIGHT_SHAPE, PRINT, REPORT_VALUE
from aslearn.parametric.mllr import MLLR
import torch as th
import numpy as np
import matplotlib.pyplot as plt
device = "cuda" if th.cuda.is_available() else "cpu"


class EDMDGPR(ExactGPR):
    
    def __init__(self, kernel: Kernel) -> None:
        super().__init__(kernel)

    def evidence(self, X:th.Tensor, Y:th.Tensor, mean_prior=None, **evidence_inputs) -> th.Tensor:
        if 'VY' not in evidence_inputs:
            raise RuntimeError("The fit function of gpr should give the variance value VY")
        VY = evidence_inputs['VY'] # (N, ny)
        N = len(VY)
        gpr_evidence = self.gpr_evidence(X, Y, mean_prior)
        noise = self.kernel.noise(X, X).repeat(1,N) # (ny, N)
         # if the regressor or hidden states is already very certain, 
         # then we don't push the noise of GPR even smaller due to numerical stability
        # if th.any(noise < 1e-9):
        #     vi_elbo = gpr_evidence
        # else:
        Sc = - 0.5*(VY.T / noise).sum(dim=1)
        vi_elbo = gpr_evidence + Sc.sum(dim=0)
        return vi_elbo
    
class GPEDMD:
    def __init__(self) -> None:
        pass
    
    def fit(self, X:th.Tensor, Xvali:th.Tensor, z_dim:int, info_level=1, max_iter=500):
        ''' traj_bundle: (traj_num, traj_len, x_dim)
            max_iter:       maximum iter optimization of the gpr, we don't need it to be very large here.
                            the opt of gpr will be warm-started each iteration.
        '''
        
        x_dim = X.shape[2]
        traj_num, traj_len = X.shape[0], X.shape[1]
        RIGHT_SHAPE(X, (traj_num,traj_len,x_dim))
        RIGHT_SHAPE(Xvali, (-1,traj_len,x_dim))
        
        # initialization
        Z_bar = th.randn(traj_num, traj_len, z_dim).double().to(device)
        VZ_bar = 1e-5*th.abs(th.randn(traj_num, traj_len, z_dim)).double().to(device)
        
        for epoch in range(2000):
            
            # fit the gp
            Z_bar_flat = Z_bar.reshape(-1, z_dim)
            VZ_bar_flat = VZ_bar.reshape(-1, z_dim)
            X_flat = X.reshape(-1, x_dim)
            
            Z_flat = Z_bar[:,:-1,:].reshape(-1, z_dim)
            Z_plus_flat = Z_bar[:,1:,:].reshape(-1, z_dim)
            VZ_flat = VZ_bar[:,:-1,:].reshape(-1, z_dim)
            VZ_plus_flat = VZ_bar[:,1:,:].reshape(-1, z_dim)
            
            kernel = RBF(x_dim, z_dim) + White(x_dim, z_dim)
            # if epoch < 3: # use 3 epoch to initialize the hidden states
            self.gpr = ExactGPR(kernel=kernel).fit(X_flat, Z_bar_flat, VY=VZ_bar_flat, info_level=info_level, max_iter=max_iter) 
            # else:
            #     self.gpr = EDMDGPR(kernel=kernel).fit(X_flat, Z_bar_flat, VY=VZ_bar_flat, info_level=info_level, max_iter=max_iter) 
            # no need the Sc? Sc term will cause digits overflow, OK seems good without Sc...

            with th.no_grad():
                self.trans = MLLR(z_dim, z_dim).fit(Z_flat, Z_plus_flat, SX=VZ_flat, SY=VZ_plus_flat, info_level=0)
                self.back = MLLR(z_dim, x_dim).fit(Z_bar_flat, X_flat, SX=VZ_bar_flat, info_level=0)
            
            # do state estimation
            with th.no_grad():
                Z_bar, VZ_bar = self.state_estimation(epoch, X, Z_bar, VZ_bar, info_level)

            # validate
            _ = self.validate(epoch, Xvali, info_level)
    
    def state_estimation(self, epoch:int, X:th.Tensor, Z_bar:th.Tensor, VZ_bar:th.Tensor, info_level:bool, 
                         his_len=200, tolerance=1e-12, penalty_factor=1):
        ''' do the state estimation
            X:      (traj_num, traj_len, x_dim)
            Z_bar:  (traj_num, traj_len, z_dim)
            VZ_bar: (traj_num, traj_len, z_dim)
            penalty_factor: An important factor to control the penalty to hidden patterns
                            the larger the heavier the penalty will be.
        '''
        x_dim, z_dim = self.gpr.kernel.input_dim, self.gpr.kernel.output_dim
        traj_len = X.shape[1]
        
        # each states will be visited maximum 20 times in expectation.
        max_iter_num = traj_len * 20
        traj_num = X.shape[0]
        
        Sigma_r_inv = th.eye(z_dim).double().to(device) * penalty_factor
        noise = self.gpr.kernel.noise(X[0,0:1,:], X[0,0:1,:]) # (z_dim, 1)
        Sigma_n_inv = 1 / noise

        Sigma_n_inv = th.diag(Sigma_n_inv.squeeze(dim=1))
        
        W = self.trans.W
        v1 = self.trans.v
        beta1 = self.trans.beta
        
        C = self.back.W
        v2 = self.back.v
        beta2 = self.back.beta
        
        # get the gp posterior
        Ef = self.gpr.posterior_mean().view(traj_num, traj_len, z_dim) # (traj_num, traj_len, z_dim)
        plot_ef = Ef[0].detach().cpu().numpy()
        plt.figure(figsize=[5,5])
        plt.plot(plot_ef)
        plt.title(f"Iter. {epoch}")
        plt.xlabel("time step")
        plt.ylabel("function value")
        plt.tight_layout()
        plt.savefig("/home/jiayun/Desktop/GPEDMD/iter_{}_post.jpg".format(epoch), dpi=150)
        plt.close()
        
        # pre-compute the covariance term
        WT_beta1_W = W.T @ beta1 @ W
        CT_beta2_C = C.T @ beta2 @ C
        
        Cov1 = th.linalg.cholesky(Sigma_n_inv + WT_beta1_W + CT_beta2_C + Sigma_r_inv)
        Cov1 = th.cholesky_inverse(Cov1)
        
        Covt = th.linalg.cholesky(Sigma_n_inv + WT_beta1_W + beta1 + CT_beta2_C + Sigma_r_inv)
        Covt = th.cholesky_inverse(Covt)
        
        CovT = th.linalg.cholesky(Sigma_n_inv + beta1 + CT_beta2_C + Sigma_r_inv)
        CovT = th.cholesky_inverse(CovT)
        
        def get_term1(Ef_t):
            term1_res = th.einsum("ij,bjk->bik", Sigma_n_inv, Ef_t.permute(0,2,1))
            return term1_res
        
        def get_term2(Z_bar_t_plus_1):
            term2_res = W.T @ beta1 # (dim_z, dim_z)
            diff_term_2 = Z_bar_t_plus_1 - v1.unsqueeze(dim=0) # (traj_num, 1, z_dim)
            diff_term_2 = diff_term_2.permute(0,2,1) # (traj_num, z_dim, 1)
            term2_res = th.einsum("ij,bjk->bik", term2_res, diff_term_2) # (traj_num, z_dim, 1)
            return term2_res
        
        def get_term3(X_t):
            term3_res = C.T @ beta2 # (z_dim, x_dim)
            diff_term_3 = X_t - v2.unsqueeze(dim=0) # (traj_num, 1, x_dim)
            diff_term_3 = diff_term_3.permute(0,2,1) # (traj_num, x_dim, 1)
            term3_res = th.einsum("ij,bjk->bik", term3_res, diff_term_3) # (traj_num, z_dim, 1)
            return term3_res
        
        def get_term4(Z_bar_t_minus_1):
            term4_res = th.einsum("ij,bjk", W, Z_bar_t_minus_1.permute(0,2,1)) + v1.unsqueeze(dim=2) # (traj_num, z_dim, 1)
            term4_res = th.einsum("ij,bjk->bik", beta1, term4_res) # (traj_num, z_dim, 1)
            return term4_res
        
        # estimation loop
        stop_flag = False
        his_variation = th.ones([his_len,]).double().to(device)
        step_counter = 0
        while not stop_flag:
            t = th.randint(low=0, high=traj_len, size=(1,))
            step_counter += 1
            
            if t == 0:
                VZ_bar[:,t,:] = th.diagonal(Cov1, dim1=0, dim2=1).unsqueeze(dim=0)
                
                term1 = get_term1(Ef[:,t:t+1,:]) # (traj_num,z_dim,1)
                term2 = get_term2(Z_bar[:,t+1:t+2,:])
                term3 = get_term3(X[:,t:t+1,:])
                
                E_z1 = term1 + term2 + term3
                E_z1 = th.einsum("ij,bjk->bik", Cov1, E_z1).squeeze(dim=2) # (traj_num, z_dim)
                variation_t = th.norm(Z_bar[:,t,:] - E_z1.unsqueeze(dim=1), dim=2).sum()
                Z_bar[:,t,:] = E_z1.unsqueeze(dim=1)
                
            elif t>0 and t<traj_len-1:
                VZ_bar[:,t,:] = th.diagonal(Covt, dim1=0, dim2=1).unsqueeze(dim=0)
                
                term1 = get_term1(Ef[:,t:t+1,:]) # (traj_num,z_dim,1)
                term2 = get_term2(Z_bar[:,t+1:t+2,:])
                term3 = get_term3(X[:,t:t+1,:])
                term4 = get_term4(Z_bar[:,t-1:t,:])

                E_z1 = term1 + term2 + term3 + term4
                E_z1 = th.einsum("ij,bjk->bik", Covt, E_z1).squeeze(dim=2) # (traj_num, z_dim)
                variation_t = th.norm(Z_bar[:,t,:] - E_z1.unsqueeze(dim=1), dim=2).sum()
                Z_bar[:,t,:] = E_z1.unsqueeze(dim=1)
                
            elif t == traj_len-1:
                VZ_bar[:,t,:] = th.diagonal(CovT, dim1=0, dim2=1).unsqueeze(dim=0)
                
                term1 = get_term1(Ef[:,t:t+1,:]) # (traj_num,z_dim,1)
                term3 = get_term3(X[:,t:t+1,:])
                term4 = get_term4(Z_bar[:,t-1:t,:])
                
                E_z1 = term1 + term3 + term4
                E_z1 = th.einsum("ij,bjk->bik", CovT, E_z1).squeeze(dim=2) # (traj_num, z_dim)
                variation_t = th.norm(Z_bar[:,t,:] - E_z1.unsqueeze(dim=1), dim=2).sum()
                Z_bar[:,t,:] = E_z1.unsqueeze(dim=1)
            
            his_variation[1:] = his_variation[0:-1].clone()
            his_variation[0] = variation_t
            mean_variation = his_variation.mean()

            if mean_variation < tolerance or step_counter > max_iter_num:
                stop_flag = True
            if step_counter % 1000 == 0:
                PRINT("-- Update step: {}, Current variation: {:.8f} --\n".format(step_counter, mean_variation.item()), info_level=info_level)
        PRINT("-- Estimation done in {} steps --\n".format(step_counter), info_level=info_level)
        
        return Z_bar, VZ_bar
    
    def forward_predict(self, x0, step_num, return_std=False):
        ''' x0:         the initial state
            step_num:   the number of forward simulation
        '''
        with th.no_grad():
            pred = [x0]
            
            z0, z0_std = self.gpr.predict(x0, return_std=True)
            z0_var = z0_std**2.
            
            W = self.trans.W
            beta1_inv = self.trans.beta_inv
            
            C = self.back.W
            beta2_inv = self.back.beta_inv

            if return_std:
                x0_var = beta2_inv + C @ th.diag(z0_var.squeeze(dim=0)) @ C.T
                var = [th.diagonal(x0_var).unsqueeze(dim=0)]
                
            for _ in range(step_num):
                z0 = self.trans.predict(z0)
                
                x0 = self.back.predict(z0)
                if return_std:
                    z0_var = beta1_inv + W @ th.diag(z0_var.squeeze(dim=0)) @ W.T
                    z0_var = th.diagonal(z0_var).unsqueeze(dim=0)
                    x0_var = beta2_inv + C @ th.diag(z0_var.squeeze(dim=0)) @ C.T
                    var.append(th.diagonal(x0_var).unsqueeze(dim=0))
                    
                pred.append(x0)

            pred = th.cat(pred)
            if return_std:
                var = th.cat(var)
                return pred, th.sqrt(var)
            return pred
        
    def validate(self, epoch,  Xvali, info_level):
        ''' validate
        '''
        traj = Xvali[0] # (traj_len, x_dim)
        T = len(traj)
        x0 = traj[0:1,:]
        
        pred, std = self.forward_predict(x0, T-1, return_std=True)
        
        pred_np = pred.detach().cpu().numpy()
        std_np = std.detach().cpu().numpy()

        t = np.arange(T)
        
        loss = th.norm(pred - traj, dim=1).sum(dim=0)
        REPORT_VALUE(loss, "Vali loss: ", info_level)
        
        plt.figure(figsize=[5,5])
        plt.plot(t, pred_np[:,0], "-b", label="pred joint angle")
        plt.plot(t, pred_np[:,1], "-c", label="pred joint vel")
        up = pred_np + std_np
        low = pred_np - std_np
        plt.fill_between(t, up[:,0], low[:,0], color='b', alpha=0.2)
        plt.fill_between(t, up[:,1], low[:,1], color='c', alpha=0.2)
        plt.plot(traj.detach().cpu().numpy(), ".r", label="GT")
        plt.title(f"Iter. {epoch}")
        plt.grid()
        plt.xlabel("time step")
        plt.ylabel("state value")
        plt.tight_layout()
        plt.savefig("/home/jiayun/Desktop/GPEDMD/iter_{0}_vali_loss_{1:.2f}.jpg".format(epoch, loss.item()), dpi=150)
        plt.close()
        return loss
        
        
if __name__ == "__main__":
    # test the gp
    # X = th.linspace(-5,5,100).reshape(-1,1).to(device).double()
    # Y = th.sin(X) + th.randn_like(X).to(device).double() * 0.2
    # VY = th.ones_like(X) * 0.1

    # kernel = RBF(1,1) + White(1,1)
    # gpr = ExpectedGPR(kernel=kernel).fit(X, Y, VY=VY, info_level=1)

    # print(kernel)   # looks good: white scale \approx 0.2^2 + 0.1

    # X = X.detach().cpu().numpy()
    # Y = Y.detach().cpu().numpy()
    # Y_pos = gpr.posterior_mean().detach().cpu().numpy()
    # import matplotlib.pyplot as plt
    # plt.plot(X, Y_pos, '-b')
    # plt.plot(X, Y, 'r.')
    # plt.show()
    
    ############################################
    # from asctr.system import Pendulum
    # from aslearn.common_utils.rollouts import collect_rollouts
    
    # P = Pendulum()
    # traj_num = 7
    # traj_len = 70
    # X_l, U_l = collect_rollouts(P, num=traj_num, traj_len=traj_len)

    # X = np.zeros((traj_num,traj_len,2))
    # for i in range(traj_num):
    #     X[i,:,:] = X_l[i]
    
    
    # vali_num = 1
    # X_l, U_l = collect_rollouts(P, num=vali_num, traj_len=traj_len)
    # Xvali = np.zeros((vali_num,traj_len, 2))
    # for i in range(vali_num):
    #     Xvali[i,:,:] = X_l[i]

    # np.save("X.npy", X)
    # np.save("Xvali.npy", Xvali)
    
    X = np.load("X.npy")
    Xvali = np.load("Xvali.npy")
    
    X = th.from_numpy(X).double().to(device)
    # X += th.randn_like(X).double().to(device) * 0.1
    Xvali = th.from_numpy(Xvali).double().to(device)
    # Xvali += th.randn_like(Xvali).double().to(device) * 0.1
    gpedmd = GPEDMD().fit(X, Xvali, 10)