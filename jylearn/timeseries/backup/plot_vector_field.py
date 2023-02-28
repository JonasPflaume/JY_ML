import casadi as cs
import torch as th
import os

import numpy as np
import matplotlib.pyplot as plt
from jycontrol.system import Pendulum
from jylearn.parametric.mlp import MLP

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica",
    'font.size': 18
})


curr_work_dir = os.path.dirname(os.path.realpath(__file__))
curr_work_dir_model = os.path.join(curr_work_dir, "models/Epoch_1000_VLoss_5.57")
Ad = th.load(curr_work_dir_model+"_A.pth")
Bd = th.load(curr_work_dir_model+"_B.pth")

hyper = {"layer":5, "nodes":[2,6,18,36,60], "actfunc":["ReLU", "ReLU", "ReLU", None]}
lifting = MLP(hyper).double()
lifting.load_model(curr_work_dir_model+"_net.pth")

p = Pendulum()
x, u = p.x_sym, p.u_sym
x_next = cs.Function("x_next", [x, u], [p.xdot])

dense = 26
x, y = np.meshgrid(np.linspace(-6.4,6.4, dense),np.linspace(-12,12, dense))
u1 = np.zeros((dense,dense))
v1 = np.zeros((dense,dense))

u2 = np.zeros((dense,dense))
v2 = np.zeros((dense,dense))

for i in range(dense):
    for j in range(dense):
        res1 = x_next([x[i,j],y[i,j]], 0.)
        u1[i,j] = res1[0]
        v1[i,j] = res1[1]
        
        input_x = th.tensor([x[i,j],y[i,j]]).double()
        xl = lifting(input_x)
        xl = th.cat([input_x, xl])
        res2_d = Ad @ xl
        res2 = ((res2_d - xl) / 0.05).detach().numpy()
        u2[i,j] = res2[0]
        v2[i,j] = res2[1]

plt.figure(figsize=[10,7])
plt.quiver(x,y,u1,v1,color='r', label="ODE")
plt.quiver(x,y,u2,v2,color='b', label="Koopman")
plt.xlabel(r"$\theta$")
plt.ylabel(r"$\dot{\theta}$")
plt.legend(loc='best')#, bbox_to_anchor=(1.,0.8))
# plt.tight_layout()
plt.savefig("/home/jiayun/Desktop/v.jpg", dpi=200)
plt.show()

### plot eigenvalues ###
eig_val, eig_vec = np.linalg.eig(Ad.detach().numpy())

# extract real part
x = [ele.real for ele in eig_val]
# extract imaginary part
y = [ele.imag for ele in eig_val]
  
# plot the complex numbers
theta = np.linspace( 0 , 2 * np.pi , 150 )
 
radius = 1.
 
a = radius * np.cos( theta )
b = radius * np.sin( theta )

plt.figure(figsize=[6,6])
plt.scatter(x, y, s=10, c='r')
plt.plot( a, b )
plt.ylabel('Imaginary')
plt.xlabel('Real')
plt.grid()
plt.tight_layout()
plt.savefig("/home/jiayun/Desktop/d.jpg", dpi=150)
plt.show()
