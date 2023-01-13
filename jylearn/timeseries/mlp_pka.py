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

class pkaDataset:
    '''
    '''
    
class PKA:
    '''
    '''

if __name__ == "__main__":
    
    from jycontrol.system import Pendulum
    
    p = Pendulum()
    X_l, U_l = collect_rollouts(p, 100, 150)
    X_lv, U_lv = collect_rollouts(p, 20, 150)
    
    trainDataset = pkaDataset(X_list=X_l, U_list=U_l)
    trainSet = data.DataLoader(dataset=trainDataset, batch_size=100, shuffle=True)
    
    valiDataset = pkaDataset(X_list=X_lv, U_list=U_lv)
    valiSet = data.DataLoader(dataset=valiDataset, batch_size=1, shuffle=False)
    
    hyper = {"layer":4, "nodes":[2,5,10,20], "actfunc":["ReLU", "ReLU", None]}
    
    mlpdmdc = PKA(hyper)
    mlpdmdc.fit(trainDataset, valiSet, lr=2e-3, epoch=1000, log_dir='/home/jiayun/Desktop/MY_ML/jylearn/timeseries/runs')