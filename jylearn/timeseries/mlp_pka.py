import torch as th
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from jylearn.parametric.mlp import MLP
from jylearn.timeseries.utils import collect_rollouts
from jylearn.timeseries.lss_em import extract_diag_block

from tqdm import tqdm
from collections import OrderedDict
device = "cuda" if th.cuda.is_available() else "cpu"
th.set_printoptions(precision=4)
#
# Convention: When we need to express the second mode of a distribution in terms of torch parameters, we mean "log precision matrices".
#

class ExternalParams(nn.Module):
    def __init__(self, dim_x, dim_obs):
        super().__init__()
        ## give it a reasonable initialization ##
        Gamma_L = th.abs(th.randn(dim_x)) * 5
        K_L = th.abs(th.randn(dim_obs)) * 5

        self.Gamma = nn.parameter.Parameter(Gamma_L).to(device).double() # process noise    - log precision matrix
        self.K = nn.parameter.Parameter(K_L).to(device).double()         # observation noise- log precision matrix
        
    def __repr__(self):
        return "Process precision matrix: {0}, Obervation precision matrix {1}".format(list(self.Gamma.shape), list(self.K.shape))

class pkaDataset(data.Dataset):
    ''' pka dataset, return the time shift dataset
    '''
    def __init__(self, X_list, U_list) -> None:
        super().__init__()
        self.size = len(X_list)
        self.generate_trajectory_triplets(X_list, U_list)
        
    def generate_trajectory_triplets(self, X_list, U_list):
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
            lr:                 float,
            epoch_num:          int,
            min_subsample_len:  int,
            logging_dir:        str
            ):
        
        net_param = PKA.setParams(self.lifting_func, 0.)
        ext_param = PKA.setParams(self.external_param, 0.)
        
        optimizer = SGD(net_param + ext_param, lr=lr, momentum=0.9)
        scheduler = ExponentialLR(optimizer, gamma=0.9)
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
                # TODO I'm working on this smoothing function
                traj_Xb_lift_smoothed, Xb_cov = PKA.batch_discrete_smoothing(self.external_param,
                                                                             A, B, C,
                                                                             traj_Xb, traj_Xb_lift, traj_Ub, start_end_index)
                
            # inner loop, update parameters by SGD
            # TODO here, modify the SGD sampling strategy !
            for _ in range(batch_num):
                optimizer.zero_grad()
                objective = PKA.log_likelihood_loss(self.external_param, A, B, C, traj_Xb_lift_smoothed, traj_Xb, traj_Ub, start_end_index)
                objective.backward()
                optimizer.step()
                
                # in order to keep A, B, C up to date, we calc them iteratively, note this step is quite cheap.
                traj_Xb_lift = self.lifting_func(traj_Xb)
                A, B, C = self.get_state_space_model(traj_Xb, traj_Ub, traj_Xb_lift)
                epoch_loss += objective.item()
                # run learning rate decay
                scheduler.step()
                
            epoch_loss /= batch_num
            
            writer.add_scalar("training loss (ELBO)", epoch_loss, global_step= epoch + 1)

            with th.no_grad():
                vali_loss, vali_fig = self.validate(vali_dataset)
            writer.add_figure("forward prediction", vali_fig, global_step= epoch + 1)
            writer.add_scalar("validation loss", vali_loss, global_step= epoch + 1)
            
            # reinitialize lr scheduler
            scheduler = ExponentialLR(optimizer, gamma=0.9)
                
    def validate(self,):
        self.lifting_func.eval()
        self.external_param.eval()
        pass
    
    def predict(self,):
        pass
    
    def predict_traj(self, x, U):
        pass
    
    @staticmethod
    def batch_discrete_smoothing(
            external_param: ExternalParams,
            A: th.Tensor,
            B: th.Tensor,
            C: th.Tensor,
            traj_Xb: th.Tensor,
            traj_Xb_lift: th.Tensor,
            traj_Ub: th.Tensor,
            start_end_index: th.Tensor):
        ''' We have to use the discrete information form to avoid running out of memory.
            Comment1:   use numba to accelerate the for loop is meaningless, because its virtue will be demolished due to
                        paralell computing.
        
            traj_Xb_lift:   (b,l,2*dim_xl)
            traj_Xb:        (b,l,dim_xl)
            traj_Ub:        (b,l-1,dim_u)
            A, B, C:        (dim_xl,dim_xl), (dim_xl,dim_u), (dim_x,dim_xl)
            start_end_index:(2,)
            
            return:
                covariance is composed of L_k, L_{k,k-1}, SER book equation (3.60)
        '''
        batch_num = traj_Xb.shape[0]
        dim_x = C.shape[0]
        dim_xl, dim_u = B.shape
        traj_length = start_end_index[1] - start_end_index[0]
        
        Gamma_precision, K_precision = th.exp(external_param.Gamma), th.exp(external_param.K)
        
        Y = traj_Xb[:,start_end_index[0]:start_end_index[1], :]
        U = traj_Ub[:,start_end_index[0]:start_end_index[1]-1, :]
        
        x0_prior = traj_Xb_lift[:, start_end_index[0], :dim_xl]
        y0 = Y[:, 0, :] # (b, dim_x)
        P0_precision_prior = th.exp(traj_Xb_lift[:, start_end_index[0], dim_xl:])
        C_t_K_inv_C = th.einsum("ij,jk->ik", C.T, th.einsum("i,ik->ik", K_precision, C))
        A_t_Gamma_inv_A = th.einsum("ij,jk->ik", A.T, th.einsum("i,ik->ik", Gamma_precision, A))
        Gamma_inv_A = th.einsum("i,ik->ik", Gamma_precision, A)
        I0 = th.diag_embed(P0_precision_prior) + C_t_K_inv_C.unsqueeze(dim=0) # (b, dim_xl, dim_xl)
        q0 = th.einsum("bi,bi->bi", P0_precision_prior, x0_prior) + th.einsum("ij,bj->bi", C.T, th.einsum("i,bi->bi", K_precision, y0))
        
        ## forward recursion ##
        D_list = []
        L_k_1_list = []
        L_k_k_1_list = []
        for i in range(traj_length-1):
            # 3.66 a.
            L_k_1_squre = I0 + A_t_Gamma_inv_A.unsqueeze(dim=0) # (b, dim_xl, dim_xl)
            L_k_1 = th.linalg.cholesky(L_k_1_squre)
            # 3.66 b.
            vk = th.einsum("ij,bj->bi", B, U[:,i,:]) # (b, dim_xl)
            rhs_3_66_b = q0 - th.einsum("ij,bj->bi", A.T, th.einsum("i,bi->bi", Gamma_precision, vk))
            d_k_1 = th.linalg.solve_triangular(L_k_1, rhs_3_66_b.unsqueeze(dim=2), upper=False).squeeze(dim=2)
            # 3.66 c.
            rhs_3_66_c = - Gamma_inv_A.unsqueeze(dim=0).repeat(batch_num, 1, 1)
            L_k_k_1 = th.linalg.solve_triangular(th.transpose(L_k_1, dim0=1, dim1=2), rhs_3_66_c, upper=True, left=False)
            # 3.66 d.
            I0 = - th.einsum("bij,bjk->bik", L_k_k_1, th.transpose(L_k_k_1, dim0=1, dim1=2)) \
                 + th.diag( Gamma_precision ).unsqueeze(dim=0).repeat(batch_num, 1, 1) \
                 + C_t_K_inv_C.unsqueeze(dim=0).repeat(batch_num, 1, 1)
            # 3.66 e.
            y0 = Y[:, i, :] # (b, dim_x)
            q0 = - th.einsum("bij,bj->bi", L_k_k_1, d_k_1) \
                 + th.einsum("i,bi->bi", Gamma_precision, vk) \
                 + th.einsum("ij,bj->bi", C.T, th.einsum("i,bi->bi", K_precision, y0))
                 
            D_list.append(d_k_1.clone())
            L_k_1_list.append(L_k_1.clone())
            L_k_k_1_list.append(L_k_k_1.clone())
        # last terms
        # 3.61 g.
        L_K_squre = I0 # (b, dim_xl, dim_xl)
        L_K = th.linalg.cholesky(L_K_squre)
        L_k_1_list.append(L_K.clone())
        # 3.63 d.
        d_K = th.linalg.solve_triangular(L_K, q0.unsqueeze(dim=2), upper=False).squeeze(dim=2)
        
        ## backward recursion ##
        # initialized x_K_pos
        x_K = th.einsum("bij,bj->bi", th.transpose(th.linalg.inv(L_K), dim0=1, dim1=2), d_K) # (b, dim_xl)
        X_smoothed_list = [x_K.unsqueeze(dim=1).clone()]
        for i in reversed( range(traj_length-1) ):
            # 3.66 f.
            L_k_1 = L_k_1_list[i]
            L_k_k_1 = L_k_k_1_list[i] # (b, dim_xl, dim_xl)
            d_k_1 = D_list[i] # (b, dim_xl)
            rhs_3_66_f = - th.einsum("bij,bj->bi", th.transpose(L_k_k_1, dim0=1, dim1=2), x_K) + d_k_1
            
            x_K = th.linalg.solve_triangular(th.transpose(L_k_1, dim0=1, dim1=2), rhs_3_66_f.unsqueeze(dim=2), upper=True).squeeze(dim=2)
            X_smoothed_list.append(x_K.unsqueeze(dim=1).clone())
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
    def get_state_space_model(traj_Xb, traj_Ub, traj_Xb_lift, padding):
        ''' solve least squares problems to get A, B, C matrices
            traj_Xb:        (b, l, dim_x)
            traj_Ub:        (b, l-1, dim_u)
            traj_Xb_lift:   (b, l, dim_xl + dim_sigma)
            padding:        padding function
        '''
        dim_xl = traj_Xb_lift.shape[2] // 2
        # dim_u = traj_Ub.shape[2]

        ### get the A,B matrices, [A,B] = V @ pinv(G)
        mb_lift, mb_lift_delay = traj_Xb_lift[:,:-1,:dim_xl], traj_Xb_lift[:,1:,:dim_xl]        # mu
        Sb_lift = traj_Xb_lift[:,:-1,dim_xl:]                                                   # Sigma
        
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
        A, B = AB[:,:dim_xl], AB[:,dim_xl:]
        
        ### get the C matrix, C = F @ pinv(L)
        F_term1_ = traj_Xb.flatten(start_dim=0, end_dim=1)                     # x_t
        F_term1 = F_term1_.unsqueeze(dim=2)
        
        F_term2_ = traj_Xb_lift[:,:,:dim_xl].flatten(start_dim=0, end_dim=1)   # \mu(x_t)
        F_term2 = F_term2_.unsqueeze(dim=2)
        F = th.einsum("bik,bjk->ij", F_term1, F_term2)
        
        L_term1 = th.einsum("bik,bjk->ij", F_term2, F_term2)
        Sb_lift_whole = traj_Xb_lift[:,:,dim_xl:].flatten(start_dim=0, end_dim=1)
        L_term2_ = 1 / th.exp(Sb_lift_whole)
        L_term2 = th.diag(L_term2_.sum(dim=0))

        L = L_term1 + L_term2
        C = F @ th.linalg.pinv(L)

        return A, B, C
    
    @staticmethod
    def log_likelihood_loss():
        pass
    
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
    
    from jycontrol.system import Pendulum
    
    p = Pendulum()
    X_l, U_l = collect_rollouts(p, 20, 150)
    X_lv, U_lv = collect_rollouts(p, 10, 150)
    
    trainDataset = pkaDataset(X_list=X_l, U_list=U_l)
    trainSet = data.DataLoader(dataset=trainDataset, batch_size=20, shuffle=True)
    
    valiDataset = pkaDataset(X_list=X_lv, U_list=U_lv)
    valiSet = data.DataLoader(dataset=valiDataset, batch_size=1, shuffle=False)
    
    dim_xl = 10
    dim_obs = 2
    
    net_hyper = {"layer":4, "nodes":[dim_obs,5,10,2*dim_xl], "actfunc":["ReLU", "ReLU", None]}
    ext_param = ExternalParams(dim_xl, dim_obs)
    mlpdmdc = PKA(net_hyper, ext_param)
    mlpdmdc.fit(trainSet, valiSet, lr=1e-2, epoch_num=1000, min_subsample_len=100, logging_dir='/home/jiayun/Desktop/MY_ML/jylearn/timeseries/runs')