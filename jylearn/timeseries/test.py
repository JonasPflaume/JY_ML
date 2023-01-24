import torch as th
from mlp_pka import PKA
import matplotlib.pyplot as plt

device = "cuda" if th.cuda.is_available() else "cpu"
th.set_printoptions(precision=4)


class ExternalParams:
    def __init__(self, dim_x, dim_obs):
        self.Gamma = th.log(th.ones(dim_x).to(device).double() * 1000)
        self.K = th.log(th.ones(dim_obs).to(device).double() * 100)
    
A = th.tensor([[0.9, -0.1],[0, 0.5]])
B = th.tensor([[0.1],[0.5]])
C = th.tensor([[0.5, 0.],[0, 0.25]])

t = th.linspace(0, 10, 200)
U = th.sin(t)
U = U[:-1].unsqueeze(dim=1)
x0 = th.ones(2,)
x0[0] = 2.
x0[1] = 1.5
X = [x0.unsqueeze(dim=0).clone()]
y0 = C @ x0 + th.randn(2,) * 0.01
Y = [y0.unsqueeze(dim=0).clone()]
for i in range(len(t)-1):
    x0 = (A @ x0.unsqueeze(dim=1) + B @ U[i].unsqueeze(dim=1)).squeeze(dim=1) + th.randn(2,) * 0.001
    y0 = (C @ x0.unsqueeze(dim=1)).squeeze(dim=1) + th.randn(2,) * 0.01
    X.append(x0.unsqueeze(dim=0).clone())
    Y.append(y0.unsqueeze(dim=0).clone())

U = U.unsqueeze(dim=0).to(device).double()
X = th.cat(X, dim=0).unsqueeze(dim=0).to(device).double()
Y = th.cat(Y, dim=0).unsqueeze(dim=0).to(device).double() # batch = 1

X = th.cat([X, th.log(th.ones_like(X).to(device).double() * 1000)], dim=2)
start_end_index = th.tensor([10, 190])
A, B, C = A.to(device).double(), B.to(device).double(), C.to(device).double()

external_param = ExternalParams(2,2)
X_smoothed, variance = PKA.batch_discrete_smoothing(external_param,
            A,
            B,
            C,
            Y,
            X,
            U,
            start_end_index)


plt.plot(X.detach().cpu().numpy().squeeze(axis=0)[10:190,:2], '-b')
plt.plot(Y.detach().cpu().numpy().squeeze(axis=0)[10:190,:2], '-k')
plt.plot(X_smoothed.detach().cpu().numpy().squeeze(axis=0), '-r')
plt.show()