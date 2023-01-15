import torch as th
from torch.optim import Adam
from torch.nn import MSELoss
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt

from jylearn.parametric.mlp import MLP
from jylearn.timeseries.utils import collect_rollouts

from tqdm import tqdm
from collections import OrderedDict
device = "cuda" if th.cuda.is_available() else "cpu"

##########
####        comment1: learning the lifting function in very high dimension directly with MLP is a bad idea !
####                  serverely overfitting!
####        comment2: numerical instability can be solved using the latest pytorch with torch.linalg.pinv
#########

class StateMatricesDataset(data.Dataset):
    ''' generate the trajectory for transition matrices fitting
    '''
    def __init__(self, X_list, U_list) -> None:
        super().__init__()
        self.size = len(X_list)
        self.generate_trajectory_triplets(X_list, U_list)
        
    def generate_trajectory_triplets(self, X_list, U_list):
        '''
        '''
        self.X_data, self.Y_data, self.U_data = [], [], []
        
        for i in range(self.size):
            X_traj, Y_traj = X_list[i][:-1,:], X_list[i][1:,:]
            U_traj = U_list[i]
            self.X_data.append(X_traj)
            self.Y_data.append(Y_traj)
            self.U_data.append(U_traj)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        X_traj, Y_traj, U_traj = self.X_data[index], self.Y_data[index], self.U_data[index]
        return X_traj, Y_traj, U_traj
    
class TripletsDataset(data.Dataset):
    ''' return the xt, ut, xt+1 triplets 
    '''
    def __init__(self, X_list, U_list) -> None:
        super().__init__()
        self.size = sum([len(X_traj)-1 for X_traj in X_list])
        self.generate_trajectory_triplets(X_list, U_list)
        
    def generate_trajectory_triplets(self, X_list, U_list):
        '''
        '''
        self.X_data, self.X_shift_data, self.U_data = [], [], []
        for traj_index in range(len(X_list)):
            X_traj, U_traj = X_list[traj_index], U_list[traj_index]
            for step_index in range(len(X_traj)-1):
                x = X_traj[step_index]
                x_shift = X_traj[step_index+1]
                u = U_traj[step_index]
                self.X_data.append(x)
                self.X_shift_data.append(x_shift)
                self.U_data.append(u)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        x, x_shift, u = self.X_data[index], self.X_shift_data[index], self.U_data[index]
        return x, x_shift, u

class MLPDMDc:
    def __init__(self, hyperparameters:dict):
        '''
        '''
        self.lifting_func = MLP(hyperparam=hyperparameters).to(device).double()
        self.neural_network_hyperparams = hyperparameters
        self.loss_func = MSELoss(reduction='mean')
        print(self.lifting_func)
        
    def fit(self, 
            matrices_dataset, 
            triplet_dataset, 
            matrices_dataset_vali,
            triplet_dataset_vali,
            lr, 
            num_epoch, 
            logging_dir):
        '''
        '''
        optimizer = Adam(self.lifting_func.parameters(), lr=lr)
        writer = SummaryWriter(logging_dir)
        
        matrices_dataset = iter(matrices_dataset)
        traj_X_batch, traj_Y_batch, traj_U_batch = next(matrices_dataset)
        traj_X_batch = traj_X_batch.to(device)
        traj_Y_batch = traj_Y_batch.to(device)
        A_dim = traj_X_batch.shape[2] + self.neural_network_hyperparams["nodes"][-1]
        B_dim = traj_U_batch.shape[2]
        self.A, self.B = th.eye(A_dim).to(device).double(), th.zeros(A_dim, B_dim).to(device).double()
        for epoch in tqdm(range(num_epoch)):
            self.lifting_func.train()
            epoch_loss = 0.0
            
            for x_batch, x_shift_batch, u_batch in triplet_dataset:
                
                # lift to latent state space
                traj_X_batch_lift = self.lifting_func(traj_X_batch)
                traj_Y_batch_lift = self.lifting_func(traj_Y_batch)
                
                traj_U_batch = traj_U_batch.to(device)
                A, B = self.get_state_matrices(traj_X_batch, traj_Y_batch, traj_X_batch_lift, traj_Y_batch_lift, traj_U_batch) # A, B should be (xl, xl) and (xl, u), not a batch data

                self.A = A
                self.B = B
                
                # batch loss simulation
                
                x_batch = x_batch.to(device)
                x_shift_batch = x_shift_batch.to(device)
                u_batch = u_batch.to(device)

                x_batch_lift = self.lifting_func(x_batch)
                x_shift_batch_lift = self.lifting_func(x_shift_batch)
                batch_loss = self.get_triplet_loss(A, B, x_batch, x_shift_batch, x_batch_lift, x_shift_batch_lift, u_batch)
                
                optimizer.zero_grad()
                batch_loss.backward()
                
                optimizer.step()
                epoch_loss += batch_loss.item()
                    
            epoch_loss /= len(triplet_dataset)
            writer.add_scalar("training loss", epoch_loss, global_step= epoch + 1)
            if epoch % 10 == 0:
                vali_loss, vali_fig = self.validate(matrices_dataset_vali, triplet_dataset_vali)
                writer.add_figure("forward prediction", vali_fig, global_step= epoch + 1)
                writer.add_scalar("validation loss", vali_loss, global_step= epoch + 1)
        
    def validate(self, matrices_dataset_vali, triplet_dataset_vali):
        '''
        '''
        with th.no_grad():
            self.lifting_func.eval()
            
            predict_trajs, gt_trajs = [], []
            validation_loss = 0.0
            ##
            for x_batch, x_shift_batch, u_batch in triplet_dataset_vali:
                # batch loss simulation
                
                x_batch = x_batch.to(device)
                x_shift_batch = x_shift_batch.to(device)
                u_batch = u_batch.to(device)

                x_batch_lift = self.lifting_func(x_batch)
                x_shift_batch_lift = self.lifting_func(x_shift_batch)
                batch_loss = self.get_triplet_loss(self.A, self.B, x_batch, x_shift_batch, x_batch_lift, x_shift_batch_lift, u_batch)
                
                validation_loss += batch_loss.item()
                    
            validation_loss /= len(triplet_dataset_vali)
            ##
            counter = 0
            for traj_X, traj_Y, traj_U in matrices_dataset_vali:
                counter += 1
                traj_X = traj_X.squeeze(dim=0).to(device)
                traj_Y = traj_Y.squeeze(dim=0).to(device)
                traj_U = traj_U.squeeze(dim=0).to(device)
                
                reconstruct_traj = self.predict_traj(traj_X[0:1,:], traj_U)
                gt_traj_tensor = th.cat([traj_X, traj_Y[-1:,:]], dim=0)

                gt_traj_ = gt_traj_tensor.detach().cpu().numpy()
                predict_trajs.append(reconstruct_traj)
                gt_trajs.append(gt_traj_)
                
                if counter == 9:
                    break
            
        # start plotting
        vali_fig = plt.figure(figsize=[10,7])
        for i in range(9):
            plt.subplot(int("33{}".format(i+1)))
            pred_traj = predict_trajs[i]
            gt_traj = gt_trajs[i]
            plt.plot(gt_traj, '-.r', label='ground truth')
            plt.plot(pred_traj, '-b', label='prediction')
            plt.grid()
            if i > 5:
                plt.xlabel("Time Step")
            if i % 3 == 0:
                plt.ylabel("States")
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.title("Vali loss: {:.2f}".format(validation_loss))
        plt.legend(by_label.values(), by_label.keys())
        plt.tight_layout()
        
        return validation_loss, vali_fig
        
    def get_state_matrices(self, traj_X_batch, traj_Y_batch, traj_X_batch_lift, traj_Y_batch_lift, traj_U_batch):
        '''
        '''
        X = traj_X_batch.view(-1, traj_X_batch.shape[-1])
        Y = traj_Y_batch.view(-1, traj_Y_batch.shape[-1])
        
        Xl = traj_X_batch_lift.view(-1, traj_X_batch_lift.shape[-1])
        Yl = traj_Y_batch_lift.view(-1, traj_Y_batch_lift.shape[-1])
        
        # add the original subspace into lifting subspace
        Xl_ = th.cat([X, Xl], dim=1)
        Yl_ = th.cat([Y, Yl], dim=1)
        U = traj_U_batch.view(-1, traj_U_batch.shape[-1])
        
        G = th.cat([Xl_, U], dim=1) # (l, dim_xl + dim_u)
        G_ = G.T @ G # (dim_xl + dim_u, dim_xl + dim_u)

        V = th.transpose(Yl_, 0, 1) @ th.cat([Xl_, U], dim=1) # Yl:(l,xl_dim) -> (xl_dim, xl_dim + u_dim)
        
        M = V @ th.linalg.pinv(G_)

        A, B = M[:, :(Xl.shape[1]+X.shape[1])], M[:, (Xl.shape[1]+X.shape[1]):]
        return A, B
    
    def get_triplet_loss(self, A, B, x_batch, x_shift_batch, x_batch_lift, x_shift_batch_lift, u_batch):
        '''
        '''
        aug_x_batch_lift = th.cat([x_batch, x_batch_lift], dim=1) # augmented lifting space
        aug_x_shift_batch_lift = th.cat([x_shift_batch, x_shift_batch_lift], dim=1)

        pred = th.einsum("ij,bj->bi", A, aug_x_batch_lift)
        pred += th.einsum("iu,bu->bi", B, u_batch)
        
        x_dim = x_batch.shape[1]
        batch_loss = self.loss_func(pred[:,:x_dim], aug_x_shift_batch_lift[:,:x_dim])
        
        return batch_loss
        
    def predict_traj(self, xinit, traj_U):
        '''
        '''
        # xinit (1, x_dim)
        xl_init = th.cat([xinit, self.lifting_func(xinit)], dim=1)
        
        reconstruct_traj = []
        reconstruct_traj.append(xinit.detach().cpu().numpy())
        # xl_init (1, x_dim + lift_dim)
        for u in traj_U:
            u = u.unsqueeze(dim=1)
            xl_init = (self.A @ xl_init.T + self.B @ u).T
            
            xinit = xl_init[:,:xinit.shape[1]]
            reconstruct_traj.append(xinit.detach().cpu().numpy())
            # lifting in a closed-loop style !
            xl_init = th.cat([xinit, self.lifting_func(xinit)], dim=1)
        return np.concatenate(reconstruct_traj)
        
if __name__ == "__main__":
    
    from jycontrol.system import Pendulum
    
    p = Pendulum()
    X_l, U_l = collect_rollouts(p, 500, 150)
    X_lv, U_lv = collect_rollouts(p, 20, 150)
    dataset1 = StateMatricesDataset(X_list=X_l, U_list=U_l)
    dataset2 = TripletsDataset(X_list=X_l, U_list=U_l)
    matrices_dataset = data.DataLoader(dataset=dataset1, batch_size=500, shuffle=True)
    triplet_dataset = data.DataLoader(dataset=dataset2, batch_size=len(dataset2), shuffle=True)
    
    dataset1v = StateMatricesDataset(X_list=X_lv, U_list=U_lv)
    dataset2v = TripletsDataset(X_list=X_lv, U_list=U_lv)
    matrices_dataset_vali = data.DataLoader(dataset=dataset1v, batch_size=1, shuffle=False)
    triplet_dataset_vali = data.DataLoader(dataset=dataset2v, batch_size=len(dataset2v), shuffle=True)
    
    hyper = {"layer":5, "nodes":[2,5,10,20,40], "actfunc":["ReLU", "ReLU", "ReLU", None]}
    
    mlpdmdc = MLPDMDc(hyper)
    mlpdmdc.fit(matrices_dataset, triplet_dataset, matrices_dataset_vali, triplet_dataset_vali, 2e-3, 1000, '/home/jiayun/Desktop/MY_ML/jylearn/timeseries/runs')