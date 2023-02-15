### JY's machine learning and control workspace, 
The **SARCOS** dataset, which is intended for a regression problem for robot inverse dynamics, is regarded as the benchmark of supervised learning algorithms.  
The gym **pendulum** environment is chosen to be the benchmark of control projects.
  
#### Finished Models
| function learning      | remark           |
| ------------- |:-------------:|
| Gaussian process regression (GPR)	|  	|
| Variational GPR (V-GPR)      |  |
| Stochastic V-GPR      |       |
| Ridge Regression      |       |

| time series      | remark           |
| ------------- |:-------------:|
|  Linear dynamical system     |   training through EM    |
|  Dynamic mode decomposition (DMD)     |   standard implementation    |
|  MLP-DMD     |   DMD with feature extraction by MLP   |
|  Differentiable Kalman smoother     |   batch implementation by tensor operation   |
|  Flexible Dynamic movement primitives (DMP)     |   Higher oder DMP   |

| Unsupervised learning      | remark           |
| ------------- |:-------------:|
|   VAE    |   Dec 14. 2022    |

Finished comparison:
#### Supervised learning  
| Approach      | MSE           |
| ------------- |:-------------:|
| VIGPR<sup>6 </sup>	|   2.879	|
| Ridge + feature      | 2.111 |
| MLP      | 1.469      |
| ... |       |
#### Control  
| Approach        | Reward Mean (50 runs) | Reward Variance|
| ------------- |:-------------:|:-------------:|
| iLQR<sup>2 </sup>      | -305.57 |   38239.82    |
| SAC<sup> 1 </sup> | -144.04 | 8377.89 |
| FLQR<sup>3 </sup> | - | - |
| DMD-MPC<sup>4 </sup> | -164.93 | 10945.07 |
| RK4-MPC<sup>5 </sup> | -142.25 | 9367.64 |
| ...| | |

Notes:  
1. In SAC(soft actor-critic) the reinforcement learning package stablebaseline3 was used to compare with various control strategies. The training ran for 50k timesteps, to a convergence of the epi reward.  
2. The cost matrices in iLQR are hard to tune by hand, and the torque constraints can't be taken into account. Therefore, its performance is just ok, but no problem to swing up and stabilize.  
3. Finite horizon LQR based on Dynamic mode decomposition(DMD). Not working, the tracking objective function is hard to define.  
4. MPC with the model learned through DMD. Very naive implementation, the symbolic feature expansion limits the speed of solution. There are better ways to include the nonlinear objective function into koopman feature.  
5. MPC with Runge Kutta 4th order simulation on ground-truth ode. This approach can't let the gym rendering in real time.  
6. The variational GPR is not suitable for inverse dynamics learning. Because there is no obvious hidden structure of this learning problem.  
