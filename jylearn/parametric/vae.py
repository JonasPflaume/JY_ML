import torch as th
import torch.nn as nn
import torch.nn.functional as F 
from torch.optim import Adam
from jylearn.parametric.mlp import MLP
import numpy as np

device = "cuda" if th.cuda.is_available() else "cpu"

class VAE(nn.Module):
    ''' variational autoencoder by simple MLP
        implementation followed:
            Doersch, C., 2016. Tutorial on variational autoencoders. arXiv preprint arXiv:1606.05908.
            
        Comment_1: predict the std is unstable, predict log_std instead !
    '''
    def __init__(self, obs_dim, latent_dim, encoder_hyperparam, decoder_hyperparam):
        super().__init__()
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim

        self.encoder = MLP(encoder_hyperparam).to(device).double()
        self.decoder = MLP(decoder_hyperparam).to(device).double()
        
    def train(self, X, lr, epoch, batch_size):
        '''
        '''
        optimizer = Adam([{"params":self.encoder.parameters()}, {"params":self.decoder.parameters()}], lr=lr)
        
        for epoch_i in range(epoch):
            epoch_loss = 0.
            b_index = th.randperm(len(X))
            batch_num = len(X)//batch_size
            for batch_i in range(batch_num):
                optimizer.zero_grad()
                
                X_b_index = b_index[batch_i*batch_size:(batch_i+1)*batch_size]
                
                X_b = X[X_b_index]
                
                elbo_loss = self.elbo(X_b)
                
                epoch_loss += elbo_loss.item()
                
                elbo_loss.backward()
                
                optimizer.step()
            
            print("== Epoch {1} Elbo Loss: {0:.2f} ==".format(epoch_loss/batch_num, epoch_i))
        
        self.encoder.eval()
        self.decoder.eval()
        
    def elbo(self, X):
        '''
        '''
        gaussian_param = self.encoder(X) # (b, 2*u)
        eps = th.normal(mean=th.zeros(self.latent_dim), std=th.ones(self.latent_dim)).to(device).double()

        mu = gaussian_param[:,:self.latent_dim]
        log_sig = gaussian_param[:,self.latent_dim:] # diagonal terms
        
        z = mu + log_sig.exp() * eps
        
        reconstruction_loss = F.binary_cross_entropy(self.decoder(z), X, reduction="sum")
        
        regularization_loss = 0.5 * ( log_sig.exp().sum(dim=1) + th.einsum("bi,bi->b", mu, mu) -\
            th.tensor(self.latent_dim).to(device).double() - th.sum(log_sig, dim=1) )

        return (reconstruction_loss + regularization_loss).sum()
        
    def sampling_from_latent_space(self):
        '''
        '''
        output_list = []
        latent_space_x = th.linspace(-3,3,10)
        latent_space_y = th.linspace(-3,3,10)
        
        for x in latent_space_x:
            for y in latent_space_y:
                x, y = x.reshape(1,-1), y.reshape(1,-1)
                X_in = th.cat([x,y],dim=1).to(device).double()
                
                output_i = self.decoder(X_in)
                output_list.append(output_i.detach().cpu().numpy())
        output_list = np.concatenate(output_list)
        return output_list
        
    def __repr__(self):
        return "Encoder: \n" + str(self.encoder) + "\n" + "Decoder: \n" + str(self.decoder)
        
if __name__ == "__main__":
    from jylearn.data.get_data import minist_data
    
    X = minist_data()
    X = X / 255.
    X = X.squeeze(axis=3)
    X = X.reshape(X.shape[0], -1)
    X = th.from_numpy(X).to(device).double()
    
    obs_dim = 28 * 28
    latent_dim = 2
    encoder_hyperparam = {"layer":5, "nodes":[obs_dim, 350, 150, 30, latent_dim*2], "actfunc":["LeakyReLU", "LeakyReLU", "LeakyReLU", None]}
    decoder_hyperparam = {"layer":5, "nodes":[latent_dim, 30, 150, 350, obs_dim], "actfunc":["LeakyReLU", "LeakyReLU", "LeakyReLU", "Sigmoid"]}
    vae = VAE(obs_dim, latent_dim, encoder_hyperparam, decoder_hyperparam)
    vae.train(X, 3e-4, epoch=50, batch_size=128)
    sampling = vae.sampling_from_latent_space()
    
    sampling = sampling.reshape(-1,28,28) * 255.
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import ImageGrid
    
    fig = plt.figure(figsize=[30,30])
    nrow, ncol = 10, 10
    grid = ImageGrid(fig, 
                 111, # as in plt.subplot(111)
                 nrows_ncols=(nrow,ncol),
                 axes_pad=0,
                 share_all=True,)
    
    i, j = 0, 0
    for col in grid.axes_column:
        for ax in col:
            imageij = sampling[10*i+j]
            ax.imshow(imageij)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            j+=1
            if j==10:
                j=0
                i+=1
    plt.show()