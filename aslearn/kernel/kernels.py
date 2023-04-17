from abc import ABC
from aifc import Error
from collections import namedtuple, OrderedDict
from copy import deepcopy
from itertools import product
import torch as th
import torch.nn as nn
import numpy as np
from enum import Enum
from functools import partial

device = "cuda" if th.cuda.is_available() else "cpu"


def next_n_alpha(s, next_n):
    ''' helper function to get the next n-th uppercase alphabet
    '''
    return chr((ord(s.upper()) + next_n - 65) % 26 + 65)

class KernelOperation(Enum):
    ''' enum class to indicate the kernel operations
    '''
    EXP = "exp"
    ADD = "add"
    MUL = "mul"

# The base class for computational graph class - Parameters 
ParametersBase = namedtuple('ParametersInfo', 
                    ("tensor_dict", "operation_dict"))

class Parameters(ParametersBase):
    
    ''' Core class to realize the operational logic of kernel combination.
        The parameters class was design to perform the kernel operation including addition, multiplication, and exponentiation.
    
        tensor_dict:        Parameters contains a dict of tensor parameters with its name.
        
        operation_dict:     The operation registration is design as a flat expansion of all operation results.
                            keys are regarded as Variables and values are the corresponding Operational Logic.
        
        The reconstruction of operations needs to evaluate the variables represented by keys in an alphabetic sequence.
    '''
    __slots__ = ()
    
    def __new__(cls, name, tensor, requres_grad=True):
        ''' we use uppercase alphabet to indicate the operation sequence
        '''
        if not isinstance(name, str):
            raise TypeError("Initialize the parameters with a string of name corresponding to the kernel.")
        if not isinstance(tensor, nn.parameter.Parameter):
            raise TypeError("We use pytorch to optimize hyperparameters.")
        
        tensor.requires_grad = requres_grad
        
        tensor_dict = OrderedDict()
        tensor_dict[name] = tensor
        
        operation_dict = OrderedDict()
        operation_dict['A'] = name # start from the root node
        return super(Parameters, cls).__new__(cls, tensor_dict, operation_dict)
    
    def join(self, param2, operation:KernelOperation):
        ''' join two parameters instance by an operation logic
            parameters will be concatenated as a unique set.
            The operator will be recoreded as a dict by steps.
        '''
        if not isinstance(operation, KernelOperation):
            raise TypeError("Use the default operation please!")
        
        param2_tensor_dict_keys = list( param2.tensor_dict.keys())

        if operation == KernelOperation.EXP:
            assert len(param2_tensor_dict_keys) == 1
            assert param2.tensor_dict[param2_tensor_dict_keys[0]].shape == (1,), "The exponentiation needs a scalar."
            assert param2.tensor_dict[param2_tensor_dict_keys[0]].requires_grad == False, \
                "The exponentiation should be a fixed parameter. Please set the requires_grad of Parameters to False."
            
        # append the operation chain from param2 to self
        max_operation_self_ori = max(self.operation_dict.keys())
        increase_n = ord(max_operation_self_ori) - 65 + 1
 
        # lift the numbering of operation dict in param2
        temp_dict = OrderedDict()
        
        for key in param2.operation_dict.keys():
            transformed_operation = [next_n_alpha(character, increase_n) if character.isupper() else character for character in param2.operation_dict[key]]
            temp_dict[next_n_alpha(key, increase_n)] =  "".join(transformed_operation)

        # concatenate operation dict
        self.operation_dict.update(temp_dict)

        # update the parameters dict
        self.tensor_dict.update(param2.tensor_dict)
        
        max_operation_self = max(self.operation_dict.keys())
        # update operation dict
        self.operation_dict[next_n_alpha(max_operation_self, 1)] = " ".join([
            max_operation_self_ori,
            operation.value,
            max_operation_self
        ])
        
    def __repr__(self) -> str:
        info =  "".join(["%"]*100) + "\n"\
                "This is a parameters group composed by: " +\
                ",".join(self.tensor_dict.keys()) + "\n" + \
                "operation numbering: \t" + ",".join(self.operation_dict.keys()) + "\n" + \
                "operations are \t" + ",".join(self.operation_dict.values()) + "\n" +\
                "".join(["%"]*100)
        return info
    
class CopiedParameter(Parameters):
    ''' deep copy the parameters group of a Parameters instance
    '''
    def __new__(cls, parameters_cls):
        operation_dict = deepcopy(parameters_cls.operation_dict)
        tensor_dict = deepcopy(parameters_cls.tensor_dict)
        return ParametersBase.__new__(cls, tensor_dict, operation_dict)
            
        
class Kernel(nn.Module):
    ''' Kernel concepts:
        1. Expand the Parameters class to register the tensor parameters
        2. forward function utilize the Parameters.operation_dict to perform kernel evaluation.
        3. operator methods were designed to update the parameters dict
    '''
    def stop_autograd(self):
        for name, param in self.named_parameters():
            if name == "exponent":
                continue
            param.requires_grad = False
    
    def start_autograd(self):
        for name, param in self.named_parameters():
            if name == "exponent":
                continue
            param.requires_grad = True
        
    def set_parameters(self, parameters_cls):
        self.curr_parameters = parameters_cls
        for name, parameter in parameters_cls.tensor_dict.items():
            self.register_parameter(name, parameter)
            
    def get_parameters(self):
        if "curr_parameters" not in self.__dict__.keys():
            raise KeyError("Parameters can only be accessed after they have been set.")
        
        return self.curr_parameters
    
    def diag(self, x):
        """ only calc the diagonal terms of k(x,x)
            NOTE: Not efficient !!!  Need to be overritten in later implementation
        """
        n = x.shape[0]
        diag_terms = []
        for i in range(n):
            term_i = self(x[i:i+1], x[i:i+1]) # (ny,1,1)
            term_i = term_i.squeeze(2).T
            diag_terms.append(term_i)
        diag_terms = th.cat(diag_terms).contiguous().T
        return diag_terms
    
    def __add__(self, right):
        if not isinstance(right, Kernel):
            raise Error("The instance should be a kernel")
        return Sum(self, right)
    
    def __radd__(self, left):
        if not isinstance(left, Kernel):
            raise Error("The instance should be a kernel")
        return Sum(left, self)
    
    def __mul__(self, right):
        if not isinstance(right, Kernel):
            raise Error("The instance should be a kernel")
        return Product(self, right)
    
    def __rmul__(self, left):
        if not isinstance(left, Kernel):
            raise Error("The instance should be a kernel")
        return Product(left, self)
    
    def __pow__(self, exponent_factor):
        if not isinstance(exponent_factor, float):
            raise Error("The instance should be a float")
        # the exponent_factor will be capsulated in a Parameters instance
        return Exponentiation(self, exponent_factor)
    
    def __repr__(self):
        return str(self._parameters)
    
class RBF(Kernel):
    ''' RBF kernel
    '''
    counter = 0
    
    def __init__(self, sigma:np.ndarray, l:np.ndarray, dim_in:int, dim_out:int):
        ''' dim_in:     the dimension of input x
            dim_out:    the dimension of output y
            l:          the kernel length should be in (dim_in, dim_out) shape
        '''
        super().__init__()
        param = np.concatenate([sigma, l.flatten()])
        param_t = th.from_numpy(param).to(device)
        self.rbf_name = "rbf{}".format(RBF.counter)
        rbf_param = nn.parameter.Parameter(param_t)
        curr_parameters = Parameters(self.rbf_name, rbf_param)

        self.set_parameters(curr_parameters)
        assert (dim_in, dim_out) == l.shape, "wrong dimension."
        assert dim_out == len(sigma), "wrong dimension."
        self.input_dim = dim_in
        self.output_dim = dim_out
        RBF.counter += 1
    
    @staticmethod
    def rbf(param, x, y, diag):
        len_x = x.shape[0] if len(x.shape)==2 else x.shape[1]
        x_dim = x.shape[1] if len(x.shape)==2 else x.shape[2]
        input_dim = x_dim
        output_dim = int( len(param) / (input_dim + 1) )

        theta = param
        sigma = theta[:output_dim].view(output_dim, 1, 1)
        l = theta[output_dim:].view(input_dim, output_dim).unsqueeze(0) # (1,nx, ny)
        
        if diag:
            assert x.shape == y.shape, "they must be the same input data!"
            distance = th.zeros(output_dim, len_x).to(device).double()
            return (sigma.squeeze(2) * th.exp( - distance ** 2 ))
        else:
            if len(x.shape)==3: # say (ny, m, nx)
                x = x.permute(1, 2, 0) # (m, nx, ny)
            else:
                x = x.view(x.shape[0], x.shape[1], 1) # (m, nx, 1)
            if len(y.shape)==3:
                y = y.permute(1, 2, 0) # (m, nx, ny)
            else:
                y = y.view(y.shape[0], y.shape[1], 1) # (m, nx, 1)
                
            x = (x/l).permute(2,0,1).contiguous() # shape: (ny, N, nx)
            y = (y/l).permute(2,0,1).contiguous() # shape: (ny, M, nx), let's regard ny axis as batch
            distance = th.cdist(x, y) # (ny, N, M)
            return sigma * th.exp( - distance ** 2 )
    
    def forward(self, x, y, diag=False):
        ''' x - (n, nx), y - (h, nx) -> (ny, n, h)
            if x or y has three axis
            such as inducing variables:
            x - (ny, m, nx), y - (h, nx) -> (ny, m, h)
        '''
        theta = eval("self.{}".format(self.rbf_name))
        return RBF.rbf(theta, x, y, diag)

class Matern(Kernel):
    ''' Matern kernel
    '''
    counter = 0
    
    def __init__(self, sigma:np.ndarray, mu:float, l:np.ndarray, dim_in:int, dim_out:int):
        super().__init__()
        param = np.concatenate([sigma, np.array([mu]), l.flatten()])
        param_t = th.from_numpy(param).to(device)
        self.matern_name = "matern{}".format(Matern.counter)
        matern_param = nn.parameter.Parameter(param_t)
        curr_parameters = Parameters(self.matern_name, matern_param)

        self.set_parameters(curr_parameters)
        assert (dim_in, dim_out) == l.shape, "wrong dimension."
        assert dim_out == len(sigma), "wrong dimension."
        self.input_dim = dim_in
        self.output_dim = dim_out
        Matern.counter += 1
    
    @staticmethod
    def matern(param, x, y, diag):
        len_x = x.shape[0] if len(x.shape)==2 else x.shape[1]
        x_dim = x.shape[1] if len(x.shape)==2 else x.shape[2]
        input_dim = x_dim
        output_dim = int( (len(param)-1) / (input_dim + 1) )
        theta = param
        sigma = theta[:output_dim].view(output_dim, 1, 1)
        mu = theta[output_dim:output_dim+1]
        l = theta[output_dim+1:].view(input_dim, output_dim).unsqueeze(0)
        
        if len(x.shape)==3: # say (ny, m, nx)
            x = x.permute(1, 2, 0) # (m, nx, ny)
        else:
            x = x.view(x.shape[0], x.shape[1], 1) # (m, nx, 1)
        if len(y.shape)==3:
            y = y.permute(1, 2, 0) # (m, nx, ny)
        else:
            y = y.view(y.shape[0], y.shape[1], 1) # (m, nx, 1)
        
        if diag:
            pass
        else:
            x = (x/l).permute(2,0,1).contiguous() # shape: (ny, N, nx)
            y = (y/l).permute(2,0,1).contiguous() # shape: (ny, M, nx), let's regard ny axis as batch
            dists = th.cdist(x, y) # (ny, N, M)
        if mu == 0.5:
            if diag:
                assert x.shape == y.shape, "they must be the same input data!"
                distance = th.zeros(output_dim, len_x).to(device).double()
                return (sigma.squeeze(2) * th.exp( - distance))
            else:
                K = sigma * th.exp(-dists)
        elif mu == 1.5:
            if diag:
                assert x.shape == y.shape, "they must be the same input data!"
                distance = th.zeros(output_dim, len_x).to(device).double()
                return (sigma.squeeze(2) * (1.0 + distance) * th.exp(-distance))
            else:
                K = dists * th.sqrt(th.tensor([3.]).to(device))
                K = sigma * (1.0 + K) * th.exp(-K)
        elif mu == 2.5:
            if diag:
                assert x.shape == y.shape, "they must be the same input data!"
                distance = th.zeros(output_dim, len_x).to(device).double()
                return (sigma.squeeze(2) * (1.0 + distance + distance**2 / 3.0) * th.exp(-distance))
            else:
                K = dists * th.sqrt(th.tensor([5.]).to(device))
                temp1 = K**2 / 3.0
                temp2 = th.exp(-K)
                K = sigma * (1.0 + K + temp1) * temp2
        elif th.isinf(mu):
            if diag:
                assert x.shape == y.shape, "they must be the same input data!"
                distance = th.zeros(output_dim, len_x).to(device).double()
                return (sigma.squeeze(2) * th.exp(-(distance**2) / 2.0))
            else:
                K = sigma * th.exp(-(dists**2) / 2.0)
        else:
            raise NotImplementedError("General cases are expensive to evaluate, please use mu = 0.5, 1.5, 2.5 and Inf")
        return K
    
    def forward(self, x, y, diag=False):
        ''' x - (n, nx), y - (h, nx) -> (ny, n, h)
            if x or y has three axis
            such as inducing variables:
            x - (ny, m, nx), y - (h, nx) -> (ny, m, h)
        '''
        theta = eval("self.{}".format(self.matern_name))
        return Matern.matern(theta, x, y, diag)

class RQK(Kernel):
    ''' rational quadratic kernel
        TODO: solve numerical instability
    '''
    counter = 0
    
    def __init__(self, sigma:np.ndarray, alpha:np.ndarray, l:np.ndarray, dim_in:int, dim_out:int):
        super().__init__()
        param = np.concatenate([sigma, alpha, l.flatten()])
        param_t = th.from_numpy(param).to(device)
        self.rqk_name = "rqk{}".format(RBF.counter)
        rqk_param = nn.parameter.Parameter(param_t)
        curr_parameters = Parameters(self.rqk_name, rqk_param)

        self.set_parameters(curr_parameters)
        assert (dim_in, dim_out) == l.shape, "wrong dimension."
        assert dim_out == len(sigma), "wrong dimension."

        self.input_dim = dim_in
        self.output_dim = dim_out
        RQK.counter += 1
    
    @staticmethod
    def rqk(param, x, y, diag):
        x_len = x.shape[0] if len(x.shape)==2 else x.shape[1]
        x_dim = x.shape[1] if len(x.shape)==2 else x.shape[2]
        input_dim = x_dim
        output_dim = int( len(param) / (input_dim + 2) )

        theta = param
        sigma = theta[:output_dim].view(output_dim, 1, 1)
        alpha = theta[output_dim:2*output_dim].view(output_dim, 1, 1)
        l = theta[2*output_dim:].view(input_dim, output_dim).unsqueeze(0) # (1,nx,ny)
        
        if len(x.shape)==3: # say (ny, m, nx)
            x = x.permute(1, 2, 0) # (m, nx, ny)
        else:
            x = x.view(x.shape[0], x.shape[1], 1) # (m, nx, 1)
        if len(y.shape)==3:
            y = y.permute(1, 2, 0) # (m, nx, ny)
        else:
            y = y.view(y.shape[0], y.shape[1], 1) # (m, nx, 1)
        if diag:
            assert x.shape == y.shape, "they must be the same input data!"
            distance = th.zeros(output_dim, x_len).to(device).double()
            return (sigma.squeeze(2) * ( 1 + distance ** 2 / alpha.squeeze(2) ) ** (-alpha.squeeze(2)))
        else:
            x = (x/l).permute(2,0,1).contiguous() # shape: (ny, N, nx)
            y = (y/l).permute(2,0,1).contiguous() # shape: (ny, M, nx), let's regard ny axis as batch
            distance = th.cdist(x, y) # (ny, N, M)
            return sigma * ( 1 + distance ** 2 / alpha ) ** (-alpha)
    
    def forward(self, x, y, diag=False):
        ''' x - (n, nx), y - (h, nx) -> (ny, n, h)
            if x or y has three axis
            such as inducing variables:
            x - (ny, m, nx), y - (h, nx) -> (ny, m, h)
        '''
        theta = eval("self.{}".format(self.rqk_name))
        return RQK.rqk(theta, x, y, diag)
    
class Constant(Kernel):
    ''' constant kernel,
        Comment1:   not recommended to use in variational methods
    '''
    counter = 0
    
    def __init__(self, c:np.ndarray, dim_in:int, dim_out:int):
        super().__init__()
        assert dim_out == len(c), "wrong dimension."
        t = th.from_numpy(c).to(device)
        self.cons_name = "cons{}".format(Constant.counter)
        cons_c = nn.parameter.Parameter(t)
        curr_parameters = Parameters(self.cons_name, cons_c)

        self.input_dim = dim_in
        self.output_dim = dim_out
        self.set_parameters(curr_parameters)
        
        Constant.counter += 1
    
    @staticmethod
    def cons(param, x, y, diag):
        output_dim = len(param)
        c = param
        c = c.view(output_dim, 1, 1)
        x_len = x.shape[0] if len(x.shape)==2 else x.shape[1]
        y_len = y.shape[0] if len(y.shape)==2 else y.shape[1]
        if diag:
            assert x.shape == y.shape, "they must be the same input data!"
            return c.squeeze(2) * th.zeros(output_dim, x_len).to(device).double()
        else:
            return c * th.ones(output_dim, x_len, y_len).to(device)
    
    def forward(self, x, y, diag=False):
        ''' x - (n, nx), y - (h, nx) -> (ny, n, h)
            if x or y has three axis
            such as inducing variables:
            x - (ny, m, nx), y - (h, nx) -> (ny, m, h)
        '''
        c = eval("self.{}".format(self.cons_name))
        return Constant.cons(c, x, y, diag)

class DotProduct(Kernel):
    ''' dot product kernel: theta * x.T @ y
        l - vector parameter is regarded as diagonal weighting matrix L
    '''
    counter = 0
    
    def __init__(self, l:np.ndarray, sigma_a:np.ndarray, sigma_b:np.ndarray, dim_in:int, dim_out:int):
        super().__init__()
        assert dim_out == l.shape[1] == len(sigma_a) == len(sigma_b), "wrong dimension."
        assert l.shape[0] == dim_in
        
        theta = np.concatenate([sigma_a, sigma_b, l.flatten()])
        t = th.from_numpy(theta).to(device)
        self.dot_name = "dot{}".format(DotProduct.counter)
        dot_l = nn.parameter.Parameter(t)
        curr_parameters = Parameters(self.dot_name, dot_l)
        
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.set_parameters(curr_parameters)
        Constant.counter += 1
    
    @staticmethod
    def dot(param, x, y, diag):
        x_dim = x.shape[1] if len(x.shape)==2 else x.shape[2]
        
        theta = param
        output_dim = int( len(param) / (x_dim + 2) )
        # (ny, 1, 1)
        sigma_a, sigma_b = theta[:output_dim].view(output_dim, 1, 1), theta[output_dim:2*output_dim].view(output_dim, 1, 1)
        l = theta[2*output_dim:].view(output_dim, x_dim).unsqueeze(1) # (ny, 1, nx)

        if diag:
            assert x.shape==y.shape, "They should be same data."
            if len(x.shape) == 3: # (ny,m,nx) (ny,h,nx) -> (ny,n,h)
                dot_res = th.einsum("ijk,ijk->ij", x-l, y-l)
            elif len(x.shape) == 2:
                x, y = x.unsqueeze(0), y.unsqueeze(0) #(1,m,nx)
                dot_res = th.einsum("ijk,ijk->ij", x-l, y-l) # (ny,m,nx)
            return sigma_a.squeeze(2) + sigma_b.squeeze(2) * dot_res
        else:
            if len(x.shape) == 3 and len(y.shape) == 2: # (ny,m,nx) (h,nx) -> (ny,m,h)
                y = y.unsqueeze(0) # (1,h,nx)
                dot_res = th.einsum("ijk,imk->ijm", x-l, y-l) # (ny,m,nx) (ny,h,nx) -> (ny,m,h)
            elif len(x.shape) == 2 and len(y.shape) == 3: # (n,nx) (ny,h,nx) -> (ny,n,h)
                x = x.unsqueeze(0) # (1,n,nx)
                dot_res = th.einsum("ijk,imk->ijm", x-l, y-l)
            elif len(x.shape) == 3 and len(y.shape) == 3: # (ny,m,nx) (ny,h,nx) -> (ny,n,h)
                dot_res = th.einsum("ijk,imk->ijm", x-l, y-l)
            else:
                x, y = x.unsqueeze(0), y.unsqueeze(0) #(1,m,nx)
                dot_res = th.einsum("ijk,imk->ijm", x-l, y-l)
            return sigma_a + sigma_b * dot_res

    def forward(self, x, y, diag=False):
        ''' x - (n, nx), y - (h, nx) -> (ny, n, h)
            if x or y has three axis
            such as inducing variables:
            x - (ny, m, nx), y - (h, nx) -> (ny, m, h)
        '''
        theta = eval("self.{}".format(self.dot_name))
        return DotProduct.dot(theta, x, y, diag)
    
class White(Kernel):
    ''' white noise kernel
    '''
    counter = 0
    
    def __init__(self, c:np.ndarray, dim_in:int, dim_out:int):
        super().__init__()
        assert dim_out == len(c), "wrong dimension."
        t = c
        t = th.from_numpy(t).to(device)
        self.white_name = "white{}".format(White.counter)
        white_c = nn.parameter.Parameter(t)
        curr_parameters = Parameters(self.white_name, white_c)
        
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.set_parameters(curr_parameters)
        White.counter += 1
        
    @staticmethod
    def white(param, x, y, diag):
        output_dim = len(param)
        c = param
        c = c.view(output_dim, 1, 1)

        x_len = x.shape[0] if len(x.shape)==2 else x.shape[1]
        y_len = y.shape[0] if len(y.shape)==2 else y.shape[1]
        if diag:
            assert x.shape == y.shape, "They should be same data."
            K = c.squeeze(2) * th.ones(output_dim, x_len).double().to(device)
        else:
            if x.shape==y.shape and th.all(x == y):
                K = c * th.eye(x_len).double().to(device).unsqueeze(0).repeat(output_dim, 1, 1)
            else:
                K = th.zeros((output_dim, x_len, y_len)).double().to(device)
        return K
    
    def forward(self, x, y, diag=False):
        ''' x - (n, nx), y - (h, nx) -> (ny, n, h)
            if x or y has three axis
            such as inducing variables:
            x - (ny, m, nx), y - (h, nx) -> (ny, m, h)
        '''
        c = eval("self.{}".format(self.white_name))
        return White.white(c, x, y, diag)
    
# there is no kernel class for exponent operation, therefore just leave it as a function
def exponent(param, x, y, diag):
    return param * 1.

class CompoundKernel(ABC):
    ''' The base class for a compound kernel
        which contains the necessary methods to 
        reconstruct the compound kernel from individual kernels.
    '''
    
    # adding new kernel needs to update this dict
    # map an operation to a static function string
    _kernel_function_dict = {
            "rbf":      "RBF.rbf",
            "cons":     "Constant.cons",
            "rqk":      "RQK.rqk",
            "white":   "White.white",
            "matern":   "Matern.matern",
            "dot":      "DotProduct.dot",
            "exponent": "exponent"
    }
    
    def generate_func_dict(self):
        ''' generate the necessary kernel functionals
        '''
        if "curr_parameters" not in self.__dict__.keys():
            raise KeyError("Parameters should have been set.")
        tensor_dict = self.curr_parameters.tensor_dict
        func_chain_dict = OrderedDict()
        for name, param in tensor_dict.items():
            command = "".join([i for i in name if not i.isdigit()])
            command = CompoundKernel._kernel_function_dict.get(command)
            command = "partial({}, param)".format(command)
            func = eval(command)
            func_chain_dict[name] = func
        return func_chain_dict
            
    def get_operation_dict(self):
        ''' get the operation logic dict
        '''
        if "curr_parameters" not in self.__dict__.keys():
            raise KeyError("Parameters should have been set.")
        operation_dict = self.curr_parameters.operation_dict
        return operation_dict
    
    def evaluate_operation_dict(self, operation_dict, x, y, diag):
        ''' calculate the numerical evaluation for each step
        '''
        operation_dict_copy = deepcopy(operation_dict)
        for key, operation in operation_dict_copy.items():
            operation = operation.split(" ")
            if len(operation) == 1: # which means basic operation
                operation_dict_copy[key] = self.func_dict[operation[0]](x, y, diag)
                
            else:
                left = operation_dict_copy[operation[0]]
                right = operation_dict_copy[operation[2]]
                if operation[1] == KernelOperation.ADD.value:
                    operation_dict_copy[key] = left + right
                elif operation[1] == KernelOperation.MUL.value:
                    operation_dict_copy[key] = left * right
                elif operation[1] == KernelOperation.EXP.value:
                    operation_dict_copy[key] = left ** right
                
        return operation_dict_copy
            
    
class Sum(Kernel, CompoundKernel):
    ''' Summation operator class
    '''
    def __init__(self, kernel1, kernel2):
        super().__init__()
        # update the parameters table and form the func table
        if not isinstance(kernel1, Kernel) or not isinstance(kernel2, Kernel):
            raise TypeError("Operands should be kernels")
        
        assert kernel1.input_dim == kernel2.input_dim, "please align the input dimenstion."
        assert kernel1.output_dim == kernel2.output_dim, "please align the input dimenstion."
        
        self.input_dim = kernel1.input_dim
        self.output_dim = kernel1.output_dim
        parameters_table1 = CopiedParameter(kernel1.get_parameters())
        parameters_table2 = kernel2.get_parameters()
        
        parameters_table1.join(parameters_table2, KernelOperation.ADD)
        
        self.set_parameters(parameters_cls=parameters_table1)
        
        self.func_dict = self.generate_func_dict()
        self.operation_dict = self.get_operation_dict()
        
    def forward(self, x, y, diag=False):
        nx = x.shape[1] if len(x.shape)==2 else x.shape[2]
        assert nx == self.input_dim, "wrong dimension."
        operation_res = self.evaluate_operation_dict(self.operation_dict, x, y, diag)
        return next(reversed(operation_res.values()))

class Product(Kernel, CompoundKernel):
    ''' Product operator class
    '''
    def __init__(self, kernel1, kernel2):
        super().__init__()
        # update the parameters table and form the func table
        if not isinstance(kernel1, Kernel) or not isinstance(kernel2, Kernel):
            raise TypeError("Operands should be kernels")
        
        assert kernel1.input_dim == kernel2.input_dim, "please align the input dimenstion."
        assert kernel1.output_dim == kernel2.output_dim, "please align the input dimenstion."
        self.input_dim = kernel1.input_dim
        self.output_dim = kernel1.output_dim
        parameters_table1 = CopiedParameter(kernel1.get_parameters())
        parameters_table2 = kernel2.get_parameters()
        
        parameters_table1.join(parameters_table2, KernelOperation.MUL)
        
        self.set_parameters(parameters_cls=parameters_table1)
        
        self.func_dict = self.generate_func_dict()
        self.operation_dict = self.get_operation_dict()
        
    def forward(self, x, y, diag=False):
        nx = x.shape[1] if len(x.shape)==2 else x.shape[2]
        assert nx == self.input_dim, "wrong dimension."
        operation_res = self.evaluate_operation_dict(self.operation_dict, x, y, diag)
        return next(reversed(operation_res.values()))
    
class Exponentiation(Kernel, CompoundKernel):
    ''' Exponentiation operator class
    '''
    def __init__(self, kernel1, exponent_factor:float):
        super().__init__()
        # update the parameters table and form the func table
        if not isinstance(kernel1, Kernel):
            raise TypeError("Operand1 should be kernels")
        if not isinstance(exponent_factor, float):
            raise TypeError("exponent_factor should be a float")
        
        self.input_dim = kernel1.input_dim
        self.output_dim = kernel1.output_dim
        parameters_table1 = CopiedParameter(kernel1.get_parameters())
        exponent_factor_t = th.tensor([exponent_factor]).to(device)
        exponent_factor_param = nn.parameter.Parameter(exponent_factor_t)
        parameters_table2 = Parameters("exponent", exponent_factor_param, requres_grad=False) # constant dosen't need grad
        
        parameters_table1.join(parameters_table2, KernelOperation.EXP)
        
        self.set_parameters(parameters_cls=parameters_table1)
        
        self.func_dict = self.generate_func_dict()
        self.operation_dict = self.get_operation_dict()
        
    def forward(self, x, y, diag=False):
        nx = x.shape[1] if len(x.shape)==2 else x.shape[2]
        assert nx == self.input_dim, "wrong dimension."
        operation_res = self.evaluate_operation_dict(self.operation_dict, x, y, diag)
        return next(reversed(operation_res.values()))