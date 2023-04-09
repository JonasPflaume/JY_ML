# The stochastic variational GPR need quite large workload to implement...
# Here I adopted the quick gp library gpytorch to show the training results
# comment: it seems the inducing variables will heavily depend on initialization,
#         the NGD methods cannot dig them out of local optimum.
#         run the code to have a look, I choose the same number of inducing variables as in vigpr.
import torch
import gpytorch
import tqdm

# inducing points should have: (num_task, m, nx) shape
class IndependentMultitaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points):
        num_tasks = inducing_points.shape[0]
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_tasks])
            )

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True
            ),
            num_tasks=num_tasks,
        )

        super().__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == "__main__":
    """ how to use
    """
    import numpy as np
    import torch as th
    import matplotlib.pyplot as plt
    device = "cuda" if th.cuda.is_available() else "cpu"
    np.random.seed(0)
    th.manual_seed(0)
    
    train_data_num = 2000 # bug? when n=100
    X = np.linspace(-20,20,100).reshape(-1,1)
    Y = np.concatenate([np.cos(X), np.sin(X)], axis=1)
    Xtrain = np.linspace(-20,20,train_data_num).reshape(-1,1)
    Ytrain1 = np.cos(Xtrain) + np.random.randn(train_data_num, 1) * 0.3 # add state dependent noise
    Ytrain2 = np.sin(Xtrain) + np.random.randn(train_data_num, 1) * 0.3
    Ytrain = np.concatenate([Ytrain1, Ytrain2], axis=1)
    Xtrain, Ytrain, X, Y = th.from_numpy(Xtrain).to(device), th.from_numpy(Ytrain).to(device),\
        th.from_numpy(X).to(device), th.from_numpy(Y).to(device)
    
    inducing_points = Xtrain[th.randperm(len(Xtrain))[:13]].unsqueeze(0).repeat(2,1,1)
    
    model = IndependentMultitaskGPModel(inducing_points).to(device).double()
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2).to(device).double()

    train_x = torch.randn(100, 4)
    train_y = torch.randn(100, 2)
    
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(Xtrain, Ytrain)
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    
    model.train()
    likelihood.train()
    
    variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=Ytrain.size(0), lr=0.1)

    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=Ytrain.size(0))

    epochs_iter = tqdm.tqdm(range(20), desc="Epoch")
    for i in epochs_iter:
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)

        for x_batch, y_batch in minibatch_iter:
            ### Perform NGD step to optimize variational parameters
            variational_ngd_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            minibatch_iter.set_postfix(loss=loss.item())
            loss.backward()
            variational_ngd_optimizer.step()
            hyperparameter_optimizer.step()
            
    model.eval()
    likelihood.eval()
    with torch.no_grad():
        preds = model(X)
        mean = preds.mean
        lower, upper = preds.confidence_region()

    X = X.detach().cpu().numpy()
    Y = Y.detach().cpu().numpy()
    mean = mean.detach().cpu().numpy()
    lower = lower.detach().cpu().numpy()
    upper = upper.detach().cpu().numpy()
    Xtrain, Ytrain = Xtrain.detach().cpu().numpy(), Ytrain.detach().cpu().numpy()
    plt.figure(figsize=[6,8])
    plt.subplot(211)
    plt.plot(X, mean[:,0], label="mean")
    plt.plot(X, upper[:,0], '-.r', label="var")
    plt.plot(X, lower[:,0], '-.r')
    plt.plot(X, Y[:,0], label="GroundTueth")
    plt.plot(Xtrain, Ytrain[:,0], 'rx', label="data", alpha=0.3)
    plt.grid()
    plt.ylabel("Output 1")
    
    plt.subplot(212)
    plt.plot(X, mean[:,1], label="mean")
    plt.plot(X, upper[:,1], '-.r', label="var")
    plt.plot(X, lower[:,1], '-.r')
    plt.plot(X, Y[:,1], label="GroundTueth")
    plt.plot(Xtrain, Ytrain[:,1], 'rx', label="data", alpha=0.3)

    plt.grid()
    plt.xlabel("Input")
    plt.ylabel("Output 2")
    plt.legend()
    plt.tight_layout()
    plt.show()