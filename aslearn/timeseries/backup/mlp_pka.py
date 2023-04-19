import torch as th
import torch.nn as nn
from torch.optim import Adam
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from aslearn.parametric.mlp import MLP
from aslearn.common_utils.rollouts import collect_rollouts

from tqdm import tqdm
from collections import OrderedDict
device = "cuda" if th.cuda.is_available() else "cpu"
th.set_printoptions(precision=4)

##
# Convention: When we need to express the second mode of a distribution in terms of torch parameters, we mean "log precision matrices".
##
# NOT Working ..., the pipeline is pathological, besides, I met the identifiability problem.
# I decided to stop here, turn to polish the theoretical stuff.
##

class ExternalParams(nn.Module):
    def __init__(self, dim_x, dim_obs):
        super().__init__()
        ## give it a reasonable initialization ##
        Gamma_L = ( th.abs(th.randn(dim_x)) * 5).to(device).double()
        K_L = (th.abs(th.randn(dim_obs)) * 5).to(device).double()
        Gamma_L.requires_grad = True
        K_L.requires_grad = True

        self.Gamma = nn.parameter.Parameter(Gamma_L) # process noise    - log precision matrix
        self.K = nn.parameter.Parameter(K_L)         # observation noise- log precision matrix
        
    def __repr__(self):
        return "Process precision matrix: {0}, Obervation precision matrix {1}".format(list(self.Gamma.shape), list(self.K.shape))

class pkaDataset(data.Dataset):
    ''' pka dataset, return the time shift dataset
    '''
    def __init__(self, X_list, U_list) -> None:
        super().__init__()
        self.size = len(X_list)
        self.generate_trajectories(X_list, U_list)
        
    def generate_trajectories(self, X_list, U_list):
        '''
        '''
        self.X_data, self.U_data = [], []
        
        for i in range(self.size):
            X_traj = X_list[i]
            U_traj = U_list[i]
            self.X_data.append(X_traj)
            self.U_data.append(U_traj)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        X_traj, U_traj = self.X_data[index], self.U_data[index]
        return X_traj, U_traj
    
    
class PKA(object):
    ''' probabilistic koopman approximation
    '''
    def __init__(self, net_hyper_param:dict, external_param) -> None:
        self.net_hyper_param = net_hyper_param
        self.lifting_func = MLP(hyperparam=net_hyper_param).to(device).double()
        self.external_param = external_param
        print(self.lifting_func)
        print(self.external_param)
    
    def fit(
            self,
            train_dataset:      pkaDataset,
            vali_dataset:       pkaDataset,
            lr_net:             float,
            lr_ext:             float,
            epoch_num:          int,
            min_subsample_len:  int,
            logging_dir:        str
            ):
        ''' Training the neural network.
        '''
        net_param = PKA.setParams(self.lifting_func, 0.)
        ext_param = [{'params': self.external_param.parameters(), 'lr':lr_ext}]#PKA.setParams(self.external_param, 0.)
        
        optimizer = Adam(net_param + ext_param, lr=lr_net)
        writer = SummaryWriter(logging_dir)
        
        whole_train_set = iter(train_dataset)
        traj_Xb, traj_Ub = next(whole_train_set) # (b, l, dim)
        traj_Xb = traj_Xb.to(device)
        traj_Ub = traj_Ub.to(device)

        # dim_x = self.net_hyper_param["nodes"][-1] // 2
        dim_u = traj_Ub.shape[2]
        traj_length = traj_Xb.shape[1]
        batch_num = traj_Xb.shape[0]
        # initialize a padding layer, used in state_space_model fitting.
        padding = nn.ZeroPad2d((0, dim_u, 0, dim_u)) # (left, right, top, bottom)
        
        # outer loop
        for epoch in tqdm(range(epoch_num)):
            self.lifting_func.train()
            self.external_param.train()
            epoch_loss = 0.0
            
            # implicit state space model
            traj_Xb_lift = self.lifting_func(traj_Xb)
            A, B, C = self.get_state_space_model(traj_Xb, traj_Ub, traj_Xb_lift, padding)

            
            # estimate belief through batch smoothing
            assert traj_length - min_subsample_len >= 0, "You should reduce the min_subsample_len smaller than trajectory length."
            start_end_index = th.randint(traj_length - min_subsample_len, size=(1,))
            start_end_index = th.cat([start_end_index, start_end_index + min_subsample_len])
            
            # batch_num is denoted as the number of batch smoothing traj
            with th.no_grad():
                traj_Xb_lift_smoothed, Xb_cov = PKA.batch_discrete_smoothing(self.external_param,
                                                                             A, B, C,
                                                                             traj_Xb, traj_Xb_lift, traj_Ub, start_end_index)
                
            # inner loop, update parameters
            for batch_index in range(50):
                optimizer.zero_grad()
                objective = PKA.log_likelihood_loss(self.external_param, A, B, C, traj_Xb_lift_smoothed, traj_Xb_lift, traj_Xb, traj_Ub, start_end_index, batch_index)
                objective.backward()
                optimizer.step()
                
                # the external parameter shouldn't be too large! reminder: covariance = 1/exp(Gamma)
                with th.no_grad():
                    self.external_param.Gamma.clamp_(-1e6, 18)
                    self.external_param.K.clamp_(-1e6, 18)
                    
                # in order to keep A, B, C up to date, we calc them iteratively, note this step is quite cheap.
                traj_Xb_lift = self.lifting_func(traj_Xb)
                A, B, C = self.get_state_space_model(traj_Xb, traj_Ub, traj_Xb_lift, padding)
                epoch_loss += objective.item()
                
            epoch_loss /= batch_num

            # I think we need to re-initialize the optimizer.
            net_param = PKA.setParams(self.lifting_func, 0.)
            ext_param = [{'params': self.external_param.parameters(), 'lr':lr_ext}]#PKA.setParams(self.external_param, 0.)

            optimizer = Adam(net_param + ext_param, lr=lr_net)
            print("Training loss: %.3f" % epoch_loss)
            print(self.external_param.K)
            writer.add_scalar("training loss (NLL)", epoch_loss, global_step= epoch + 1)

            # start validation
            with th.no_grad():
                # update saved state space
                self.A, self.B, self.C = A.clone(), B.clone(), C.clone()
                vali_loss, vali_fig = self.validate(vali_dataset)
            writer.add_figure("forward prediction", vali_fig, global_step= epoch + 1)
            writer.add_scalar("validation loss", vali_loss, global_step= epoch + 1)
                
    def validate(self, vali_dataset):
        ''' start validation
        '''
        self.lifting_func.eval()
        self.external_param.eval()
        
        dim_xl, dim_u = self.B.shape
        dim_x = self.C.shape[0]
        dim_xl -= dim_x
        L_func = nn.MSELoss()
        
        vali_loss = 0.
        X_list, pred_X_list, pred_var_list = [], [], []
        with th.no_grad():
            for X, U in vali_dataset:
                X, U = X.squeeze(dim=0).to(device).double(), U.squeeze(dim=0).to(device).double()
                
                x0 = X[0:1, :] # (1,dim_x)
                x0_lift = self.lifting_func(x0)
                x0_lift_prior, P0_lift_prior = x0_lift[:,:dim_xl].T, x0_lift[:,dim_xl:].T # (dim_xl, 1)
                x0_lift_prior = th.cat([x0.T, x0_lift_prior], dim=0)
                P0_lift_prior = th.cat([th.ones(dim_x, 1).to(device).double() * 15, P0_lift_prior], dim=0)
                
                traj_X, traj_var = self.predict_traj(x0_lift_prior, P0_lift_prior, U)
                
                vali_loss += L_func(traj_X, X).item()
                pred_X_list.append(traj_X.clone())
                pred_var_list.append(traj_var.clone())
                X_list.append(X)
        
        vali_loss /= len(vali_dataset)
        
        vali_fig = PKA.plot_vali_traj(X_list, pred_X_list, pred_var_list, vali_loss)
        return vali_loss, vali_fig
            
            
    @staticmethod
    def plot_vali_traj(X_list, pred_X_list, pred_var_list, vali_loss):
        ''' plot the first 9 trajectories
        '''
        
        vali_fig = plt.figure(figsize=[10,7])
        for i in range(9):
            plt.subplot(int("33{}".format(i+1)))
            pred_traj = pred_X_list[i].detach().cpu().numpy()
            pred_var = pred_var_list[i].detach().cpu().numpy()
            gt_traj = X_list[i].detach().cpu().numpy()
            t = np.arange(len(pred_traj))
            plt.plot(t, gt_traj, '-.r', label='ground truth')
            plt.plot(t, pred_traj, '-b', label='prediction')
            # plt.fill_between(t, pred_traj[:,0]+pred_var[:,0], pred_traj[:,0]-pred_var[:,0], facecolor='green', alpha=0.3)
            # plt.fill_between(t, pred_traj[:,1]+pred_var[:,1], pred_traj[:,1]-pred_var[:,1], facecolor='green', alpha=0.3)
            plt.grid()
            if i > 5:
                plt.xlabel("Time Step")
            if i % 3 == 0:
                plt.ylabel("States")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.title("Vali loss: {:.2f}".format(vali_loss))
        plt.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        return vali_fig
    
    def predict_traj(self, x0_lift_prior, P0_lift_prior, U):
        ''' forward simulation, A \mu + B u, A P A.T + Gamma
        '''
        with th.no_grad():
            P0 = 1/th.exp(P0_lift_prior)
            Gamma = 1/th.exp(self.external_param.Gamma)
            K = 1/th.exp(self.external_param.K)
            
            yt = self.C @ x0_lift_prior
            var_t = self.C @ th.diag(P0.squeeze(dim=1)) @ self.C.T + th.diag(K)
            
            traj_X = [yt.T.clone()]
            traj_var = [th.diag(var_t).unsqueeze(dim=0).clone()]
            for i in range(len(U)):
                ut = U[i:i+1,:]
                
                x0_lift_prior = self.A @ x0_lift_prior + self.B @ ut.T
                P0 = th.diag(self.A @ th.diag(P0.squeeze(dim=1)) @ self.A.T + th.diag(Gamma)).unsqueeze(dim=1)
                
                yt = self.C @ x0_lift_prior
                var_t = self.C @ th.diag(P0.squeeze(dim=1)) @ self.C.T + th.diag(K)
                
                traj_X.append(yt.T.clone())
                traj_var.append(th.diag(var_t).unsqueeze(dim=0).clone())
                
            traj_X = th.cat(traj_X, dim=0)
            traj_var = th.cat(traj_var, dim=0)
        
        return traj_X, traj_var
        
    
    @staticmethod
    def batch_discrete_smoothing(
            external_param:             ExternalParams, 
            A:                          th.Tensor,          # (dim_xl,dim_xl)
            B:                          th.Tensor,          # (dim_xl,dim_u)
            C:                          th.Tensor,          # (dim_x,dim_xl)
            traj_Xb:                    th.Tensor,          # (b,l,dim_xl)
            traj_Xb_lift:               th.Tensor,          # (b,l,2*dim_xl)
            traj_Ub:                    th.Tensor,          # (b,l-1,dim_u)
            start_end_index:            th.Tensor           # (2,)
            ):
        ''' We have to use the discrete information form to avoid running out of memory.
            The batch discrete smoothing was implemented through "information form" of Kalman filter + smoother.
            Reference:  "State estimation for robotics." BARFOOT.
            
            Comment1:   use numba to accelerate the for loop is meaningless, because its virtue will be demolished due to
                        paralell computing.
            return:
                covariance is composed of L_k, L_{k,k-1}, SER book equation (3.60)
        '''
        batch_num = traj_Xb.shape[0]
        dim_x = traj_Xb.shape[2]
        dim_xl = traj_Xb_lift.shape[2] // 2
        traj_length = start_end_index[1] - start_end_index[0]
        
        ## parameters initialization checked
        Gamma_precision, K_precision = th.exp(external_param.Gamma), th.exp(external_param.K)
        
        Y = traj_Xb[:,start_end_index[0]:start_end_index[1], :]
        U = traj_Ub[:,start_end_index[0]:start_end_index[1]-1, :]
        
        x0_prior = th.cat([traj_Xb[:, start_end_index[0], :], traj_Xb_lift[:, start_end_index[0], :dim_xl]], dim=1)
        y0 = Y[:, 0, :] # (b, dim_x)
        P0_precision_prior_ = th.cat([th.ones(batch_num, dim_x).to(device).double() * 15, traj_Xb_lift[:, start_end_index[0], dim_xl:]], dim=1)
        P0_precision_prior = th.exp(P0_precision_prior_) # (b, dim_xl)
        ##
        
        ## forward loop initialization checked
        C_t_K_inv_C = th.einsum("ij,jk->ik", C.T, th.einsum("i,ik->ik", K_precision, C))
        A_t_Gamma_inv_A = th.einsum("ij,jk->ik", A.T, th.einsum("i,ik->ik", Gamma_precision, A))
        Gamma_inv_A = th.einsum("i,ik->ik", Gamma_precision, A)
        # 3.67 a.
        I0 = th.diag_embed(P0_precision_prior) + C_t_K_inv_C.unsqueeze(dim=0) # (b, dim_xl, dim_xl)
        # 3.67 b.
        q0 = th.einsum("bi,bi->bi", P0_precision_prior, x0_prior) + th.einsum("ij,bj->bi", C.T, th.einsum("i,bi->bi", K_precision, y0))
        ##
        
        ## forward recursion ##
        D_list = []
        L_k_1_list = []
        L_k_k_1_list = []
        for i in range(traj_length-1):
            # 3.66 a. # checked
            L_k_1_squre = I0 + A_t_Gamma_inv_A.unsqueeze(dim=0) # (b, dim_xl, dim_xl)
            L_k_1 = th.linalg.cholesky(L_k_1_squre)
            # 3.66 b. # checked
            vk = th.einsum("ij,bj->bi", B, U[:,i,:]) # (b, dim_xl)
            rhs_3_66_b = q0 - th.einsum("ij,bj->bi", A.T, th.einsum("i,bi->bi", Gamma_precision, vk)) # (b, dim_xl)
            d_k_1 = th.linalg.solve_triangular(L_k_1, rhs_3_66_b.unsqueeze(dim=2), upper=False).squeeze(dim=2)
            # 3.66 c. # checked
            rhs_3_66_c = - Gamma_inv_A.unsqueeze(dim=0).repeat(batch_num, 1, 1)
            L_k_k_1 = th.linalg.solve_triangular(th.transpose(L_k_1, dim0=1, dim1=2), rhs_3_66_c, upper=True, left=False)
            # 3.66 d. # checked
            I0 = - th.einsum("bij,bjk->bik", L_k_k_1, th.transpose(L_k_k_1, dim0=1, dim1=2)) \
                 + th.diag( Gamma_precision ).unsqueeze(dim=0).repeat(batch_num, 1, 1) \
                 + C_t_K_inv_C.unsqueeze(dim=0).repeat(batch_num, 1, 1)
            # 3.66 e. # checked
            y0 = Y[:, i+1, :] # (b, dim_x)
            q0 = - th.einsum("bij,bj->bi", L_k_k_1, d_k_1) \
                 + th.einsum("i,bi->bi", Gamma_precision, vk) \
                 + th.einsum("ij,bj->bi", C.T, th.einsum("i,bi->bi", K_precision, y0))
                 
            D_list.append(d_k_1.clone())
            L_k_1_list.append(L_k_1.clone())
            L_k_k_1_list.append(L_k_k_1.clone())
        # last terms
        # 3.61 g. # checked
        L_K_squre = I0 # (b, dim_xl, dim_xl)
        L_K = th.linalg.cholesky(L_K_squre)
        L_k_1_list.append(L_K.clone())
        # 3.63 d. # checked
        d_K = th.linalg.solve_triangular(L_K, q0.unsqueeze(dim=2), upper=False).squeeze(dim=2)
        
        ## backward recursion ## # checked
        # initialized x_K_pos
        x_K = th.einsum("bij,bj->bi", th.transpose(th.linalg.inv(L_K), dim0=1, dim1=2), d_K) # (b, dim_xl)
        X_smoothed_list = [x_K.unsqueeze(dim=1).clone()]
        for i in reversed( range(traj_length-1) ):
            # 3.66 f.
            L_k_1 = L_k_1_list[i] # (b, dim_xl, dim_xl)
            L_k_k_1 = L_k_k_1_list[i] # (b, dim_xl, dim_xl)
            d_k_1 = D_list[i] # (b, dim_xl)
            
            rhs_3_66_f = - th.einsum("bij,bj->bi", th.transpose(L_k_k_1, dim0=1, dim1=2), x_K) + d_k_1
            x_K = th.linalg.solve_triangular(th.transpose(L_k_1, dim0=1, dim1=2), rhs_3_66_f.unsqueeze(dim=2), upper=True).squeeze(dim=2)
            X_smoothed_list.append(x_K.unsqueeze(dim=1).clone() )
        X_smoothed_list.reverse()
        X_smoothed = th.cat(X_smoothed_list, dim=1)
        
        cov0 = th.einsum("bij,bkj->bik", L_k_1_list[0], L_k_1_list[0]) # (b, dim_xl, dim_xl)
        variance_list = [th.diagonal(cov0, dim1=1, dim2=2).unsqueeze(dim=1).clone()]

        for i in range( 1, traj_length ):
            cov0 = th.einsum("bij,bkj->bik", L_k_1_list[i], L_k_1_list[i]) + th.einsum("bij,bkj->bik", L_k_k_1_list[i-1], L_k_k_1_list[i-1])
            variance_list.append(th.diagonal(cov0, dim1=1, dim2=2).unsqueeze(dim=1).clone())
        variance = th.cat(variance_list, dim=1) # (b, l, dim_xl)

        return X_smoothed, variance
    
    @staticmethod
    def get_state_space_model(
            traj_Xb,                # (b, l, dim_x)
            traj_Ub,                # (b, l-1, dim_u)
            traj_Xb_lift,           # (b, l, dim_xl + dim_sigma)
            padding                 # padding function
            ):
        ''' solve least squares problems to get A, B, C matrices
            Checked.
        '''
        dim_xl = traj_Xb_lift.shape[2] // 2
        batch_num, length, dim_x = traj_Xb.shape
        # dim_u = traj_Ub.shape[2]

        ### get the A,B matrices, [A,B] = V @ pinv(G)
        mb_lift, mb_lift_delay = traj_Xb_lift[:,:-1,:dim_xl], traj_Xb_lift[:,1:,:dim_xl]        # mu
        mb_lift = th.cat([traj_Xb[:,:-1,:], mb_lift], dim=2)
        mb_lift_delay = th.cat([traj_Xb[:,1:,:], mb_lift_delay], dim=2)
        Sb_lift = traj_Xb_lift[:,:-1, dim_xl:]                                                  # Sigma
        Sb_lift = th.cat([th.ones(batch_num,length-1,dim_x).to(device).double() * 15, Sb_lift], dim=2)

        ## batch outer product to get V
        # concatenate data
        V_term1_ = mb_lift_delay.flatten(start_dim=0, end_dim=1)
        V_term1 = V_term1_.unsqueeze(dim=2)
        V_term2_ = th.cat([mb_lift, traj_Ub], dim=2).flatten(start_dim=0, end_dim=1)
        V_term2 = V_term2_.unsqueeze(dim=2)
        V = th.einsum("bik,bjk->ij", V_term1, V_term2)
        
        ## batch outer product to get G
        G_term1 = th.einsum("bik,bjk->ij", V_term2, V_term2)
        # concatenate data
        G_term2_1 = Sb_lift.flatten(start_dim=0, end_dim=1)
        G_term2_2 = 1 / th.exp(G_term2_1) # mark: output of network is - log precision matrix
        G_term2_3 = th.sum(G_term2_2, dim=0)
        G_term2_4 = th.diag(G_term2_3)
        G_term2 = padding(G_term2_4)
        G = G_term1 + G_term2
        AB = V @ th.linalg.pinv(G)
        A, B = AB[:,:dim_xl+dim_x], AB[:,dim_xl+dim_x:]
        
        ### get the C matrix, C = F @ pinv(L)
        # F_term1_ = traj_Xb.flatten(start_dim=0, end_dim=1)                     # x_t
        # F_term1 = F_term1_.unsqueeze(dim=2)
        
        # F_term2_ = traj_Xb_lift[:,:,:dim_xl].flatten(start_dim=0, end_dim=1)   # \mu(x_t)
        # F_term2 = F_term2_.unsqueeze(dim=2)
        # F = th.einsum("bik,bjk->ij", F_term1, F_term2)
        
        # L_term1 = th.einsum("bik,bjk->ij", F_term2, F_term2)
        # Sb_lift_whole = traj_Xb_lift[:,:,dim_xl:].flatten(start_dim=0, end_dim=1)
        # L_term2_ = 1 / th.exp(Sb_lift_whole)
        # L_term2 = th.diag(L_term2_.sum(dim=0))

        # L = L_term1 + L_term2
        # C = F @ th.linalg.pinv(L)
        C = th.zeros(dim_x, dim_x + dim_xl).to(device).double()
        for i in range(dim_x):
            C[i,i] = 1.

        return A, B, C
    
    @staticmethod
    def log_likelihood_loss(external_param:         ExternalParams, # external parameter class
                            A:                      th.Tensor,      # A matrix 
                            B:                      th.Tensor,      # B matrix
                            C:                      th.Tensor,      # C matrix
                            traj_Xb_lift_smoothed:  th.Tensor,      # belief state
                            traj_Xb_lift:           th.Tensor,      # lifting states, this is used to extract prior state
                            traj_Xb:                th.Tensor,      # observation
                            traj_Ub:                th.Tensor,      # control
                            start_end_index:        th.Tensor,      # two elements indicate the start and end index
                            batch_index:            th.Tensor,      # every update in SGD, we evaluate only one trajectory
                            ):
        ''' log likelihood loss
            This function was written in a form to utilize the parallel computing with GPU, therefore, quite efficient to compute
            the whole trajectory.
        '''
        dim_x, dim_xl = traj_Xb.shape[2], traj_Xb_lift.shape[2]//2
        batch_num = traj_Xb.shape[0]
        X_smoothed = traj_Xb_lift_smoothed
        U = traj_Ub[:, start_end_index[0]:start_end_index[1]-1, :]
        X_obs = traj_Xb[:, start_end_index[0]:start_end_index[1], :]
        
        x0_prior = traj_Xb_lift[:, start_end_index[0]:start_end_index[0]+1, :dim_xl]
        x0_prior = th.cat([traj_Xb[:, start_end_index[0]:start_end_index[0]+1, :], x0_prior], dim=2)
        P0_prior = traj_Xb_lift[:, start_end_index[0], dim_xl:]
        P0_prior = th.cat([th.ones(batch_num, dim_x).to(device).double() * 15, P0_prior], dim=1)
        nll = 0.
        
        ### check step 1
        mean_1 = x0_prior
        P0_prior_precision = th.exp(P0_prior)
        
        x0_rvar = X_smoothed[:,0:1,:] # random variable
        temp1 = th.einsum("bj,blj->blj", th.sqrt(P0_prior_precision), x0_rvar - mean_1) # (b,dim_xl) x (b,1,dim_xl) -> (b, 1, dim_xl)
        temp2 = 0.5 * th.log(th.prod(P0_prior_precision, dim=1, keepdim=True)) - 0.5 * th.norm(temp1, dim=2) ** 2.
        nll -= temp2.sum()
        # because the magnitude of initial state loss will generally len(X_smoothed) times smaller than other 2 terms
        ###
        
        ### check step 2
        mean_2_ = th.einsum("ij,blj->bli", A, X_smoothed[:,:-1,:]) # (b,l-1,dim_xl)
        mean_2 = mean_2_ + th.einsum("ij,blj->bli", B, U)
        Gamma_precision = th.exp(external_param.Gamma)

        x_rvar = X_smoothed[:,1:,:] # random variable
        temp3 = th.einsum("j,blj->blj", th.sqrt(Gamma_precision), x_rvar - mean_2) # (b,l-1,dim_xl)
        temp4 = 0.5 * th.log(th.prod(Gamma_precision)) - 0.5 * th.norm(temp3, dim=2) ** 2.
        nll -= temp4.sum()
        ###
        
        ### check step 3
        mean_3 = th.einsum("ij,blj->bli", C, X_smoothed) # (b,l,dim_xl)
        K_precision = th.exp(external_param.K)

        x_rvar_2 = X_obs # random variable
        temp5 = th.einsum("j,blj->blj", th.sqrt(K_precision), x_rvar_2 - mean_3)
        temp6 = 0.5 * th.log(th.prod(K_precision)) - 0.5 * th.norm(temp5, dim=2) ** 2.
        nll -= temp6.sum()
        ###
        
        return nll
    
    @staticmethod
    def setParams(network:nn.Module, decay:float) -> list:
        ''' function to set weight decay
        '''
        params_dict = dict(network.named_parameters())
        params=[]

        for key, value in params_dict.items():
            if key[-4:] == 'bias':
                params += [{'params':value,'weight_decay':0.0}]
            else:
                params +=  [{'params': value,'weight_decay':decay}]
        return params

if __name__ == "__main__":
    
    from asctr.system import Pendulum
    
    p = Pendulum()
    X_l, U_l = collect_rollouts(p, 100, 200)
    X_lv, U_lv = collect_rollouts(p, 10, 200)
    
    trainDataset = pkaDataset(X_list=X_l, U_list=U_l)
    trainSet = data.DataLoader(dataset=trainDataset, batch_size=100, shuffle=True)
    
    valiDataset = pkaDataset(X_list=X_lv, U_list=U_lv)
    valiSet = data.DataLoader(dataset=valiDataset, batch_size=1, shuffle=False)
    
    dim_xl = 40
    dim_x = 2
    
    net_hyper = {"layer":5, "nodes":[dim_x,5,15,45,2*dim_xl], "actfunc":["ReLU", "ReLU", "ReLU", None]}
    ext_param = ExternalParams(dim_x+dim_xl, dim_x)
    mlpdmdc = PKA(net_hyper, ext_param)
    mlpdmdc.fit(trainSet, valiSet, lr_net=1e-2, lr_ext=1e-2, epoch_num=500, min_subsample_len=150, logging_dir='/home/jiayun/Desktop/MY_ML/jylearn/timeseries/runs')