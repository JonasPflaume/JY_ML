from base64 import decode
from torch.optim import Adam
from torch.nn import MSELoss
import torch as th
import numpy as np
from jylearn.parametric.mlp import MLP

device = "cuda" if th.cuda.is_available() else "cpu"

class Endecoder:
    '''
    '''
    def __init__(self, param_encoder, param_decoder) -> None:
        self.encoder = MLP(param_encoder).to(device)
        self.decoder = MLP(param_decoder).to(device)
    
    def fit(self, Xtrain, Xval, lr=1e-3, batch_size=64, steps=2000):
        Loss = MSELoss()
        param = list(self.encoder.parameters()) + list(self.decoder.parameters())
        optimizer = Adam(param, lr=lr)
        Xtrain_t = th.from_numpy(Xtrain).to(device).float()
        Xval_t = th.from_numpy(Xval).to(device).float()
        counter = 0
        for _ in range(steps):
            counter += 1
            optimizer.zero_grad()
            b_index = np.random.choice(np.arange(len(Xtrain)), batch_size, replace=True)
            X_bat = Xtrain_t[b_index]
            encode_X = self.encoder(X_bat)
            decode_X = self.decoder(encode_X)
            
            Loss_b = Loss(X_bat, decode_X)
            Loss_b.backward()
            optimizer.step()
            
            if counter % 500 == 0:
                with th.no_grad():
                    self.decoder.eval(), self.encoder.eval()
                    encode_X = self.encoder(Xval_t)
                    decode_X = self.decoder(encode_X)
                    Loss_b = Loss(Xval_t, decode_X)
                    Loss_b = Loss_b.detach().cpu().numpy()
                    print("Current val Loss: ", Loss_b)
                    
                    self.decoder.train(), self.encoder.train()
                    
        self.decoder.eval(), self.encoder.eval()
        
    def load(self):
        pass
    
    def __call__(self, x):
        
        x = th.from_numpy(x).to(device).float()
        
        return self.encoder(x).detach().cpu().numpy()
    
#param = {"layer":4, "nodes":[21, 500, 500, 7], "batch":128, "lr":1e-3, "decay":0.}
#net = MLP(param).to(device)

if __name__ == "__main__":
    # test fit
    mean = [0, 0]
    cov = [[1, 0], [0, 10]]  # diagonal covariance
    X = np.random.multivariate_normal(mean, cov, 200)
    Xval = np.random.multivariate_normal(mean, cov, 50)
    
    param1 = {"layer":3, "nodes":[2, 5, 20]}
    param2 = {"layer":3, "nodes":[20, 5, 2]}
    
    import matplotlib.pyplot as plt

    ed = Endecoder(param1, param2)
    ed.fit(X, Xval)
    
    x = th.from_numpy(Xval).to(device).float()
        
    xpred = ed.decoder(ed.encoder(x)).detach().cpu().numpy()
    
    plt.plot(Xval[:,0], Xval[:,1], '.r')
    plt.plot(xpred[:,0], xpred[:,1], '.b')
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    plt.show()
