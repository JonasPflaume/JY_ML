from aslearn.base.regression import Regression
import torch as th
from torch.optim import LBFGS, Adam
import numpy as np
from scipy.optimize import minimize
device = "cuda" if th.cuda.is_available() else "cpu"
th.pi = th.acos(th.zeros(1)).item() * 2

class ExactGPR(Regression):
    
    def __init__(self, kernel):
        '''
            kernel:
                    the kernel function
        '''
        self.kernel = kernel
        self.evidence_evaluate_counter = 0

    def fit(self, X, Y, 
            call_hyper_opt=True, 
            solver="LBFGS",
            lbfgs_evidence_barier=-1e9,
            adam_batch_size=128,
            adam_tolerance=1e-5,
            adam_lr=5e-3,
            verbose=False):
        '''
        '''
        self.kernel.start_autograd()
        
        if len(X.shape) == 1:
            # in case the data is 1-d
            X = X.reshape(-1,1)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1,1)
        n = X.shape[0]
        d = X.shape[1]
        m = Y.shape[1] # output dimension
        with th.no_grad():
            self._X = X
        # call the hyperparameter optimization
        
        if call_hyper_opt:
            evidence = float("-inf")
            self.kernel.train()
            if solver == "LBFGS":
                optimizer = LBFGS(params=self.kernel.parameters(), 
                                    lr=1, 
                                    max_iter=1000, 
                                    tolerance_change=1e-12,
                                    line_search_fn="strong_wolfe")
                
                self.curr_evidence = evidence
                def closure():
                    self.kernel.guarantee_non_neg_param()
                            
                    self.evidence_evaluate_counter += 1
                    optimizer.zero_grad()
                    try:
                        evidence = ExactGPR.evidence(self.kernel, X, Y)
                        objective = - evidence
                        objective.backward()
                    except:
                        evidence = th.tensor(lbfgs_evidence_barier)
                        objective = -evidence
                    self.curr_evidence = evidence.item()
                    if verbose:
                        if self.evidence_evaluate_counter % 50 == 0:
                            print("Current evidence: %.2f" % evidence.item())
                        else:
                            print(">", end="")
                    return objective
                
                optimizer.step(closure)
                self.kernel.guarantee_non_neg_param()
                
            elif solver == "Adam":
                optimizer = Adam(params=self.kernel.parameters(), lr=adam_lr)
                self.curr_evidence = evidence
                loss_history = [float("inf")]
                stop_flag = False
                while not stop_flag:
                    optimizer.zero_grad()
                    batch_index = th.randperm(len(X))[:adam_batch_size]
                    evidence = ExactGPR.evidence(self.kernel, X[batch_index], Y[batch_index])
                    loss = - evidence
                    self.evidence_evaluate_counter += 1
                    loss.backward()
                    optimizer.step()
                        
                    if self.evidence_evaluate_counter % 100 == 0 and verbose:
                        print("Step: ", self.evidence_evaluate_counter, "Current evidence: %.2f" % evidence.item())
                    
                    self.kernel.guarantee_non_neg_param()
                            
                    loss_history.append(loss.item())
                    if len(loss_history) > 300:
                        loss_history.pop(0)
                        
                        if np.abs(np.mean(loss_history[:150]) - np.mean(loss_history[150:])) < adam_tolerance:
                            stop_flag = True
                
            self.kernel.eval()
            self.kernel.stop_autograd()
        # output guarantee for kernel postive param
        self.kernel.guarantee_non_neg_param()
        K = self.kernel(X, X) # K has shape (ny, n, n)
        try:
            u = th.linalg.cholesky(K) # u has shape (ny, n, n)
        except:
            if verbose:
                print("The cho_factor meet singular matrix, now add damping...")
            K += th.eye(K.shape[1]).unsqueeze(0).repeat(m, 1, 1).to(device) * 1e-6
            u = th.linalg.cholesky(K) # u has shape (ny, n, n)
        
        Y = Y.permute(1, 0)
        Y = Y.unsqueeze(2)
        self.alpha = th.cholesky_solve(Y, u) # (ny, n, 1) solve ny linear system independently
        self.L = u # shape (ny, n, n)
        
        # print the margianl likelyhood
        # Y (ny, n, 1), alpha (ny, n, 1), L (ny, n, n)
        diagonal_term = th.log(th.diagonal(self.L, dim1=1, dim2=2)).sum(dim=1)
        evidence = - 0.5 * th.einsum("ijk,ijk->i", Y, self.alpha) - diagonal_term - n / 2 * th.log(2*th.tensor([th.pi]).to(device))
        evidence = evidence.sum(axis=-1)
        if verbose:
            print("The evidence is: %.2f" % evidence.item())
        del K, u, evidence
        th.cuda.empty_cache()
        return self
        
    @staticmethod
    def evidence(kernel, X, Y, channel_process="sum"):
        n = X.shape[0]
        m = Y.shape[1] # output dimension

        K = kernel(X, X) # K has shape (ny, n, n)
        try:
            u = th.linalg.cholesky(K) # u has shape (ny, n, n)
        except:
            K = K + th.eye(K.shape[1]).unsqueeze(0).repeat(m, 1, 1).to(device) * 1e-6
            u = th.linalg.cholesky(K) # u has shape (ny, n, n)
            
        Y = Y.permute(1, 0)
        Y = Y.unsqueeze(2)

        alpha = th.cholesky_solve(Y, u) # (ny, n, 1) solve ny linear system independently
        L = u # shape (ny, n, n)
        
        # print the margianl likelyhood
        # Y (ny, n, 1), alpha (ny, n, 1), L (ny, n, n)
        diagonal_term = th.log(th.diagonal(L, dim1=1, dim2=2)).sum(dim=1)
        evidence = - 0.5 * th.einsum("ijk,ijk->i", Y, alpha) - diagonal_term - n / 2 * th.log(2*th.tensor([th.pi]).to(device))
        if channel_process == "sum":
            evidence = evidence.sum(axis=-1)
        elif channel_process == "mean":
            evidence = evidence.mean(axis=-1)

        return evidence
    
    def predict(self, x, return_var=False):
        '''
        '''
        if len(x.shape) == 1:
            # in case the data is 1-d
            x = x.reshape(-1,1)
            
        k = self.kernel(x, self._X) # (ny, n_*, n), alpha: (ny, n, 1)
        mean = th.einsum("ijk,ikb->ji", k, self.alpha)
        if return_var:
            k = k.permute(0,2,1)
            v = th.linalg.solve_triangular(self.L, k, upper=False) # (ny, n, n_*)
            v = v.permute(2, 0, 1) # (n_*, ny, n)
            prior_std = self.kernel(x,x,diag=True) # the diag in kernel base class changed. (ny, n_*)
            var = prior_std.T - th.einsum("ijk,ijk->ij", v, v)
            white_noise = self.kernel.white_diag(x)
            var += white_noise
            del k, v, prior_std
            th.cuda.empty_cache()
            return mean, var.reshape(-1,self.kernel.output_dim)
        else:
            del k
            th.cuda.empty_cache()
            return mean
        
    def maximum_entropy_point(self, box_cons, K=10):
        ''' treat the prediction as a diagonal multivariate gaussian distribution
            we then maximize its entropy
        '''
        
        def maximum_entropy_obj(x):
            x_f = x.reshape(1,-1)
            X_t = th.from_numpy(x_f).to(device).double()
            _, var = self.predict(X_t, return_var=True)
            entropy = th.sum(th.log(var[0,:])).detach().cpu().numpy()
            return -entropy
            
        x_his = []
        f_his = []
        bounds = []
        for i in range(box_cons.shape[1]):
            bounds.append((box_cons[0,i], box_cons[1,i]))
        bounds = tuple(bounds)
        for _ in range(K):
            # print("Hi!")
            x0 = np.random.uniform(box_cons[0], box_cons[1])
            res_i = minimize(fun=maximum_entropy_obj, x0=x0, method='L-BFGS-B', bounds=bounds)
            
            x_his.append(res_i.x)
            f_his.append(res_i.fun)
            
        opt_index = np.argmin(f_his)
        x_opt = x_his[opt_index]
        return x_opt
        
if __name__ == "__main__":
    # test
    import matplotlib.pyplot as plt
    from aslearn.kernel.kernels import RBF, White, Matern, DotProduct, RQK, Constant
    import numpy as np
    from torch.nn import MSELoss
    Loss = MSELoss()
    np.random.seed(0)
    th.manual_seed(0)
    
    l = np.ones([1, 2]) * 1.5
    c = np.array([0.1, 0.1])
    
    kernel = White(c=c, dim_in=1, dim_out=2) + RBF(l=l, dim_in=1, dim_out=2)
    gpr = ExactGPR(kernel=kernel)
    
    train_data_num = 90 # bug? when n=100
    X = np.linspace(-10,10,100).reshape(-1,1)
    Y = np.concatenate([np.cos(X), np.sin(X)], axis=1)
    Xtrain = np.linspace(-10,10,train_data_num).reshape(-1,1)
    Ytrain1 = np.cos(Xtrain) + np.random.randn(train_data_num, 1) * 0.3 # add state dependent noise
    Ytrain2 = np.sin(Xtrain) + np.random.randn(train_data_num, 1) * 0.4
    Ytrain = np.concatenate([Ytrain1, Ytrain2], axis=1)
    Xtrain, Ytrain, X, Y = th.from_numpy(Xtrain).to(device), th.from_numpy(Ytrain).to(device),\
        th.from_numpy(X).to(device), th.from_numpy(Y).to(device)
    
    # train
    gpr.fit(Xtrain, Ytrain, verbose=True)
    print( gpr.kernel )
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
    plt.fill_between(X.squeeze(), mean[:,0]-var[:,0], mean[:,0]+var[:,0], color='b', alpha=0.2)
    plt.plot(X, Y[:,0], label="GroundTueth")
    plt.plot(Xtrain, Ytrain[:,0], 'rx', label="data", alpha=0.4)
    plt.grid()
    plt.ylabel("Output 1")
    
    plt.subplot(212)
    plt.plot(X, mean[:,1], label="mean")
    plt.fill_between(X.squeeze(), mean[:,1]-var[:,1], mean[:,1]+var[:,1], color='b', alpha=0.2)
    plt.plot(X, Y[:,1], label="GroundTueth")
    plt.plot(Xtrain, Ytrain[:,1], 'rx', label="data", alpha=0.4)
    plt.grid()
    plt.xlabel("Input")
    plt.ylabel("Output 2")
    plt.legend()
    plt.tight_layout()
    plt.show()