from collections import namedtuple
import torch as th
import torch.nn as nn

class Parameters(
                    namedtuple('ParametersInfo', 
                    ("name", "dim", "tensor", "bounds"))
                ):
    
    __slots__ = ()
    
    def __new__(cls, name, dim, tensor, bounds):
        
        if not isinstance(name, str):
            raise TypeError("Initialize the parameters with a string of name corresponding to the kernel.")
        if not isinstance(dim, int):
            raise TypeError("The dimension of parameters should be a integer.")
        if not isinstance(tensor, nn.parameter.Parameter):
            raise TypeError("We use pytorch to optimize hyperparameters.")
        if not isinstance(bounds, th.Tensor):
            raise TypeError("We use pytorch to optimize hyperparameters.")
        
        assert len(tensor) == dim, "false dimension or tensor."
        assert len(bounds.shape) == 2, "second dimension for upper and lower bounds."
        assert bounds.shape[0] == dim, "align the dimension please."
        assert bounds.shape[1] == 2, "Lower and upper bounds."
        
        return super(Parameters, cls).__new__(cls, name, dim, tensor, bounds)
        
class Kernel(nn.Module):
    
    def forward(self, x):
        pass
    
    def __add__(self, right):
        pass
    
    def __radd__(self, left):
        pass
    
    def __mul__(self, right):
        pass
    
    def __rmul__(self, left):
        pass
    
    def __pow__(self, exponent_factor):
        pass
    
    def __repr__(self):
        pass
    
    def _check_bounds(self):
        pass
    
class Sum(nn.modules):
    pass

class Product(nn.modules):
    pass

class Exponentiation(nn.Module):
    pass


class RBF(Kernel):
    pass

class Matern(Kernel):
    pass

class RQK(Kernel):
    pass

class Period(Kernel):
    pass

class Constant(Kernel):
    pass

class InnerProduct(Kernel):
    pass
