from aslearn.base.regression import Regression
import torch as th
import numpy as np
from torch.optim import LBFGS
from tqdm import tqdm
import copy
device = "cuda" if th.cuda.is_available() else "cpu"
th.pi = th.tensor([th.acos(th.zeros(1)).item() * 2]).to(device)

# def setParams(network:th.nn.Module) -> list:
#         ''' function to set weight decay
#         '''
#         params_dict = dict(network.named_parameters())
#         params=[]

#         for key, value in params_dict.items():
#             if key[-4:] == 'bias':
#                 params += [{'params':value}]
#             else:
#                 params +=  [{'params': value}]
#         return params

class VariationalEMSparseGPR(Regression):
    
    def __init__(self, kernel, white_kernle):
        '''
            Following the paper of Titsias (2009)
            
            Comment 1: EM VIGPR will over-fit !
            
            kernel:
                    the kernel function
            white_kernel:
                    the white kernel should be treated differently
                    because the channel noise have to be treated in a different way than the exact GPR
        '''
        self.kernel = kernel
        self.white_kernel = white_kernle
        
    def fit(self, X, Y, m, 
            subsetNum=50, 
            lr=1e-2,
            stop_criterion=1e-4,
            no_max_step=False,
            no_exp_step=False):
        '''
            steps:
                    1. e step pick point (for each channel?)
                    2. m step maximize elbo via optimize w.r.t. theta
                    3. when done, fit the u and A of \phi(f_m) (if for each channel then u and A have shape (ny, m), (ny, m, m))
                    
                    no_max_step:    the inducing variable will be choose first, then optimize the hyper parameters
        '''
        if len(X.shape) == 1:
            # in case the data is 1-d
            X = X.reshape(-1,1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1,1)

        self.kernel.train()
        self.white_kernel.train()
        inducing_var_index = [[] for _ in range(Y.shape[1])] # 2d list, each entry represents a output channel
        if no_exp_step:
            for j in range(len(inducing_var_index)):
                inducing_var_index[j] = np.random.choice(np.arange(len(X)), m, replace=False).tolist()
        else:
            curr_mean_elbo = -float("inf")
            pbar = tqdm(range(m), desc="Current mean elbo: {}".format(curr_mean_elbo))
            for i in pbar:
                # expectation step
                next_inducing_var = VariationalEMSparseGPR.rank_inducing_points(self.kernel, self.white_kernel, X, Y, inducing_var_index, subsetNum)
                # next_inducing_var should be a list with length ny
                for j in range(len(inducing_var_index)):
                    
                    inducing_var_index[j].append(next_inducing_var[j])
                
                # maximization step
                if not no_max_step:
                    curr_mean_elbo = self.hyper_optimization(X, Y, inducing_var_index, lr, stop_criterion)
                
                pbar.set_description("Current mean elbo: {:.2f}".format(curr_mean_elbo))
                th.cuda.empty_cache()
        
        if no_max_step or no_exp_step:
            curr_mean_elbo = self.hyper_optimization(X, Y, inducing_var_index, lr, stop_criterion)
            print("Current elbo: {:.2f}".format(curr_mean_elbo))

        # prepare parameters for the prediction
        self.sigma = dict(self.white_kernel.named_parameters())[self.white_kernel.white_name].view(-1,1,1) # (ny,1,1)
        Xm = []
        for i in range(Y.shape[1]):
            Xm_y_index = inducing_var_index[i]
            Xm_y = X[Xm_y_index].unsqueeze(0)
            Xm.append(Xm_y)
        Xm = th.cat(Xm, dim=0) # (ny, m, nx)
        Kmn = self.kernel(Xm, X) # (ny, m, n)
        Knm = Kmn.permute(0,2,1) # (ny, n, m)
        Kmm = self.kernel(Xm, Xm) + th.eye(m,m).to(device).double() * 1e-6 # (ny, m, m)
        
        Sigma = Kmm + 1/self.sigma * th.bmm(Kmn, Knm)
        u = th.linalg.cholesky(Sigma)
        
        term1_mu = th.cholesky_solve(th.einsum("ijk,ki->ij", Kmn, Y).unsqueeze(2), u).squeeze(2) # (ny,m) ?
        term2_A = th.cholesky_solve(Kmm, u) # (ny,m,m)
        mu = 1/self.sigma.squeeze(2) * th.einsum("ijk,ik->ij", Kmm, term1_mu) # (ny,m)
        self.Kmm_inv = th.linalg.cholesky(Kmm)
        self.Kmm_inv = th.cholesky_solve(th.eye(m).to(device).double(), self.Kmm_inv) # (ny,m,m)
        self.Kmm_inv_mu = th.einsum("ijk,ik->ij", self.Kmm_inv, mu) #(ny,m)
        A = th.einsum("ijk,ikb->ijb", Kmm, term2_A) # (ny,m,m)
        self.Kmm_inv_A_Kmm_inv = th.einsum("ijk,ikb,ibm->ijm", self.Kmm_inv, A, self.Kmm_inv)
        self.Xm = Xm

        self.kernel.eval()
        self.white_kernel.eval()
        return inducing_var_index
        
    def predict(self, x, return_var=False):
        ''' 
            use the distribution \phi(f_m) and 
            equation (8) to generate the predictive distribution
        '''
        Kxm = self.kernel(x, self.Xm) # (ny, *, m)
        Kmx = Kxm.permute(0,2,1) # (ny, m, *)
        
        mean = th.einsum("ijk,ik->ij", Kxm, self.Kmm_inv_mu) # (ny,m)
        
        if return_var:
            Kxx = self.kernel(x, x, diag=True) # (ny,*)
            term1 = th.einsum("ijk,ikb->ijb", Kxm, self.Kmm_inv) # (ny,*,m)
            term1 = th.einsum("ijb,ibj->ij", term1, Kmx)
            
            term2 = th.einsum("ijk,ikb->ijb", self.Kmm_inv_A_Kmm_inv, Kmx)
            term2 = th.einsum("ijk,ikj->ij", Kxm, term2)
            var = Kxx - term1 - term2
            var = var + self.sigma.view(-1,1)
            return mean.T, var.T
        return mean.T
    
    def hyper_optimization(self, X, Y, inducing_var_index, lr, stop_criterion):
        ''' maximization step
        '''
        Xm = []
        for i in range(Y.shape[1]):
            Xm_y_index = inducing_var_index[i]
            Xm_y = X[Xm_y_index].unsqueeze(0)
            Xm.append(Xm_y)
            
        Xm = th.cat(Xm, dim=0) # (ny, m, nx)

        param = list(self.kernel.parameters()) + list(self.white_kernel.parameters())
        
        optimizer = LBFGS(params=param, lr=lr, max_iter=40, tolerance_change=stop_criterion)
                
        self.elbo_sum = -float("inf")
        def closure():
            self.kernel.guarantee_non_neg_param()
            self.white_kernel.guarantee_non_neg_param()
            optimizer.zero_grad()
            curr_elbo = VariationalEMSparseGPR.elbo(self.kernel, self.white_kernel, X, Y, Xm)
            curr_elbo = curr_elbo.sum()
            self.elbo_sum = curr_elbo.item()
            objective = -curr_elbo
            objective.backward()

            return objective
        
        optimizer.step(closure)
        self.kernel.guarantee_non_neg_param()
        self.white_kernel.guarantee_non_neg_param()
                
        return self.elbo_sum # return the mean elbo
                
    @staticmethod
    def elbo(kernel, white_kernel, X, Y, Xm):
        ''' two way function:
                1. the greed selection function for e step, output a vector with shape (ny,n-m)
                2. the loss of m step, output a vector with shape (ny,)
                
            X:  (n, nx)
            Y:  (n, ny)
            Xm: (ny, m, nx)
        '''
        n, m = X.shape[0], Xm.shape[1]
        output_noise = dict(white_kernel.named_parameters())[white_kernel.white_name].view(-1,1,1) # (ny,1,1)

        F0 = - n/2 * th.log(2*th.pi) - (n-m)/2 * th.log(output_noise.squeeze()) - 1/(2*output_noise.squeeze()) * th.einsum("ij,ij->j", Y, Y)

        Kmm = kernel(Xm, Xm) # (ny, m, m)
        F1 = 0.5 * th.log( th.diagonal(Kmm, dim1=1, dim2=2) ).sum(dim=1)

        Kmn= kernel(Xm, X)
        Knm = Kmn.permute(0,2,1)
        Kmn_Knm = th.bmm(Kmn, Knm) # (ny,m,m)
        Term1 = output_noise * Kmm + Kmn_Knm
        
        F2 = -0.5 * th.log( th.diagonal(Term1, dim1=1, dim2=2) ).sum(dim=1)

        # there is any problem with F3 term
        try:
            u = th.linalg.cholesky(Term1) # (ny,m,m)
        except:
            Term1 = Term1 + th.eye(m,m).to(device).double() * 1e-6
            u = th.linalg.cholesky(Term1)
            
        b = th.einsum("ijk,ki->ij", Kmn, Y) # (ny,m)
        inv_Term1 = th.cholesky_solve(b.unsqueeze(2), u).squeeze(2) # (ny,m)

        leading_term = th.einsum("ij,jik->jk", Y, Knm) # (ny,m)
        F3 = 1/(2*output_noise.squeeze()) * th.einsum("ij,ij->i", leading_term, inv_Term1)

        diag_terms = kernel(X, X, diag=True)
        F4 = - 1/(2*output_noise.squeeze()) * th.sum(diag_terms, dim=1)
        
        try:
            u = th.linalg.cholesky(Kmm) # (ny,m,m)
        except:
            Kmm = Kmm + th.eye(m,m).to(device).double() * 1e-6
            u = th.linalg.cholesky(Kmm)
            
        inside_trace = th.cholesky_solve(Kmn_Knm, u) # (ny,m,m)
        F5 = 1/(2*output_noise.squeeze()) * inside_trace.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)

        Fv = F0 + F1 + F2 + F3 + F4 + F5
        return Fv

    @staticmethod
    def rank_inducing_points(kernel, white_kernel, X, Y, inducing_var_index, subsetNum) -> list:
        ''' expectation step
            inducing_var_index: A list has ny lists.
            subsetNum:          the maximum evaluating subset
            
            1. generate a 2-d index
            2. each time evaluate the elbo for one dimension
            3. calc and return the maximum index (ny,) list
        '''
        if subsetNum > len(X):
            subsetNum = len(X)
        # candidates list
        candi_list = np.array([np.random.choice(np.setdiff1d(np.arange(len(X)), np.array(inducing_var_index[i])), subsetNum, replace=False).tolist() \
            for i in range(Y.shape[1])], dtype=int)# numpy (ny, candi_num)
        candi_scores = np.zeros_like(candi_list.T)
        candi_list_copy = copy.deepcopy(candi_list.tolist())
        candi_list = candi_list.T.tolist() # list (candi_num, ny)
        for j in range(len(candi_list)):
            candi_i_index = candi_list[j]
            Xm = []
            for i in range(Y.shape[1]):
                Xm_y_index = inducing_var_index[i] + [candi_i_index[i]]
                Xm_y = X[Xm_y_index].unsqueeze(0)
                Xm.append(Xm_y)
            Xm = th.cat(Xm, dim=0) # (ny, m, nx)
            elbo_score = VariationalEMSparseGPR.elbo(kernel, white_kernel, X, Y, Xm).detach().cpu().numpy() # score should be (ny,) numpy array
            candi_scores[j,:] = elbo_score

        index_max = np.argmax(candi_scores, axis=0)
        next_index = []
        for j in range(Y.shape[1]):
            next_index.append(candi_list_copy[j][index_max[j]])
        return next_index

if __name__ == "__main__":
    # test
    import matplotlib.pyplot as plt
    from aslearn.kernel.kernels import RBF, White, Matern, DotProduct, RQK, Constant
    import numpy as np
    from torch.nn import MSELoss
    Loss = MSELoss()
    np.random.seed(0)
    th.manual_seed(0)
    
    l = np.ones([1, 2]) * 1.
    c = np.array([0.3, 0.3])
    
    white_kernel = White(c=c, dim_in=1, dim_out=2)
    kernel = RBF(l=l, dim_in=1, dim_out=2)
    
    gpr = VariationalEMSparseGPR(kernel=kernel, white_kernle=white_kernel)
    
    train_data_num = 1000 # bug? when n=100
    X = np.linspace(-20,20,100).reshape(-1,1)
    Y = np.concatenate([np.cos(X), np.sin(X)], axis=1)
    Xtrain = np.linspace(-20,20,train_data_num).reshape(-1,1)
    Ytrain1 = np.cos(Xtrain) + np.random.randn(train_data_num, 1) * 0.3 # add state dependent noise
    Ytrain2 = np.sin(Xtrain) + np.random.randn(train_data_num, 1) * 0.3
    Ytrain = np.concatenate([Ytrain1, Ytrain2], axis=1)
    Xtrain, Ytrain, X, Y = th.from_numpy(Xtrain).to(device), th.from_numpy(Ytrain).to(device),\
        th.from_numpy(X).to(device), th.from_numpy(Y).to(device)
    
    # train
    ind = gpr.fit(Xtrain, Ytrain, m=13, subsetNum=200, lr=1e-2)
    
    import time
    s = time.time()
    for i in range(50):
        mean, var = gpr.predict(X, return_var=True)
    e = time.time()
    print("The time for each pred: %.5f" % ((e-s)/100))
    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    mean = mean.detach().cpu().numpy()
    var = var.detach().cpu().numpy()
    Xtrain, Ytrain = Xtrain.detach().cpu().numpy(), Ytrain.detach().cpu().numpy()
    plt.figure(figsize=[6,8])
    plt.subplot(211)
    plt.plot(X, mean[:,0], label="mean")
    plt.plot(X, mean[:,0] + 1.96*var[:,0], '-.r', label="var")
    plt.plot(X, mean[:,0] - 1.96*var[:,0], '-.r')
    plt.plot(X, Y[:,0], label="GroundTueth")
    plt.plot(Xtrain[ind[0]], Ytrain[ind[0],0], 'c*')
    plt.plot(Xtrain, Ytrain[:,0], 'rx', label="data", alpha=0.3)
    plt.grid()
    plt.ylabel("Output 1")
    
    plt.subplot(212)
    plt.plot(X, mean[:,1], label="mean")
    plt.plot(X, mean[:,1] + 1.96*var[:,1], '-.r', label="var")
    plt.plot(X, mean[:,1] - 1.96*var[:,1], '-.r')
    plt.plot(X, Y[:,1], label="GroundTueth")
    plt.plot(Xtrain[ind[1]], Ytrain[ind[1], 1], 'c*')
    plt.plot(Xtrain, Ytrain[:,1], 'rx', label="data", alpha=0.3)

    plt.grid()
    plt.xlabel("Input")
    plt.ylabel("Output 2")
    plt.legend()
    plt.tight_layout()
    plt.show()