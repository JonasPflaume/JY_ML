import torch as th
import torch.nn as nn

# Addition, multiplication, exponentiation should be possible
# RBF, Polynomial, Constant, Matern kernels

class Kernel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def __definition(self, x):
        raise NotImplementedError