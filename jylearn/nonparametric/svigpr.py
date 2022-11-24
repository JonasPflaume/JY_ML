# Gpytorch implementation
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

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([num_tasks]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_tasks])),
            batch_shape=torch.Size([num_tasks])
        )

    def forward(self, x):
        # The forward function should be written as if we were dealing with each output
        # dimension in batch
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

if __name__ == "__main__":
    inducing_points = torch.randn(2, 16, 4)
    model = IndependentMultitaskGPModel(inducing_points)
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=2)

    train_x = torch.randn(100, 4)
    train_y = torch.randn(100, 2)
    
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    
    model.train()
    likelihood.train()
    
    variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=0.1)

    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=0.01)
    
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    epochs_iter = tqdm.tqdm(range(50), desc="Epoch")
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
        preds = model(train_x)
        means = preds.mean.cpu()