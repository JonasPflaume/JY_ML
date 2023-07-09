"""
Compoundable kernel class.

This implementation enables product *, summation +, and exponentiation ** operations
to build up new kernels from fundamental kernels.
- use parameters() to get all the pytorch parameters.
- noise() method is designed to obtain the white noise, call pytorch forward method 
to evaluate all the kernels except white kernel. This will be comfortable when implementing the GPs.
- white noise kernel can't be used in product operation.
- use get_parameters() to check the compounding operation logic list.
- softplus was used to guarantee non-negative kernel parameters
"""

from abc import ABC
from aifc import Error
from collections import namedtuple, OrderedDict
from copy import deepcopy
from itertools import product
import torch as th
import torch.nn as nn
from torch.nn.functional import softplus
import numpy as np
from enum import Enum
from functools import partial
from typing import Union, Optional
from aslearn.common_utils.check import RIGHT_SHAPE

device = "cuda" if th.cuda.is_available() else "cpu"

SOFT_PLUS_BETA = 20.
EPSILON_MARGIN = 1e-10
THRESHOLD_OUT = softplus(th.tensor(20./SOFT_PLUS_BETA)).item()

def next_n_alpha(s:str, next_n:int):
    ''' helper function to get the next n-th uppercase alphabet
    '''
    return chr((ord(s.upper()) + next_n - 65) % 26 + 65)

def lower_bounded_tensor(param:th.nn.Parameter) -> th.nn.Parameter:
    ''' x = softplus(param) + epsilon, epsilon small positive number
    '''
    return softplus(param, beta=SOFT_PLUS_BETA) + EPSILON_MARGIN

def inverse_softplus(x:th.Tensor) -> th.Tensor:
    ''' inverse softplus function,
        no need to call very frequently, we can use the THRESHOLD_OUT to check
        if the softplus function are in the saturation region.
        
        the input value shouldn't be too 'negative', otherwise there will be numerical error.
    '''
    prior_res = th.zeros_like(x).double().to(device)
    linear_mask = x > THRESHOLD_OUT
    prior_res[linear_mask] = x[linear_mask]
    softplus_mask = th.logical_not(linear_mask)
    prior_res[softplus_mask] = 1/SOFT_PLUS_BETA * \
    th.log(th.exp((x[softplus_mask]-EPSILON_MARGIN)*SOFT_PLUS_BETA)-1.)
    assert not th.any(th.isinf(prior_res)), "the orgininal value is too 'negative' ..."
    return prior_res

# x = th.tensor([-1, 1, 1e-8, -100,100.]).double().to(device)
# t = lower_bounded_tensor(x)
# print(t)
# print(inverse_softplus(t))

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
    
    def __new__(cls, name:str, tensor:nn.parameter.Parameter, requres_grad=True):
        ''' we use uppercase alphabet to indicate the operation sequence,
            because usually the kernel combination won't beyond 26 times.
            name:       naming your parameter
            tensor:     the data tensor
        '''
        if not isinstance(name, str):
            raise TypeError("Initialize the parameters with a string of name corresponding to the kernel.")
        if not isinstance(tensor, nn.parameter.Parameter):
            raise TypeError("We use pytorch to optimize hyperparameters.")
        
        tensor.requires_grad = requres_grad
        
        tensor_dict = OrderedDict()
        tensor_dict[name] = tensor
        
        operation_dict = OrderedDict()
        operation_dict['A'] = name # start from the root node A
        return super(Parameters, cls).__new__(cls, tensor_dict, operation_dict)
    
    def join(self, param2, operation:KernelOperation) -> None:
        ''' join two parameters instance by an operation logic: KernelOperation
            parameters will be concatenated as a unique set.
            The operator will be recoreded as a dict by steps.
            param2:     another Parameters instance
            operation:  KernelOperation
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
        ''' print the class
        '''
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
    def __init__(self) -> None:
        super().__init__()
        self.input_dim = None
        self.output_dim = None
        
    def stop_autograd(self) -> None:
        ''' set all learnable parameters to requires_grad = False
        '''
        for name, param in self.named_parameters():
            if name == "exponent":
                continue
            param.requires_grad = False
    
    def start_autograd(self) -> None:
        ''' set all learnable parameters to requires_grad = True
        '''
        for name, param in self.named_parameters():
            if name == "exponent":
                continue
            param.requires_grad = True
        
    def set_parameters(self, parameters_cls:nn.parameter.Parameter) -> None:
        ''' when creating an exact kernel use this method to register the parameters to pytorch backend
            parameters_cls:     pytorch Parameters
        '''
        self.curr_parameters = parameters_cls
        for name, parameter in parameters_cls.tensor_dict.items():
            self.register_parameter(name, parameter)
            
    def get_parameters(self) -> nn.parameter.Parameter:
        ''' get current parameters
        '''
        if "curr_parameters" not in self.__dict__.keys():
            raise KeyError("Parameters can only be accessed after they have been set.")
        
        return self.curr_parameters
    
    def diag(self, x:th.Tensor) -> th.Tensor:
        """ only calc the diagonal terms of k(x,x)
            x:          input, (N, nx)
            output:     (ny, N)
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
    
    def noise(self, x:th.Tensor, y:th.Tensor, diag=False) -> th.Tensor:
        ''' white noise should be calc independently,
            but I want to integrate it under the same class
            no input, directly return the noise parameters
            
            x,y,diag act only as place-holder.
        '''
        ny = self.output_dim
        total_white_noise = 0.
        for name, param in self.named_parameters():
            if "white" in name:
                RIGHT_SHAPE(param, (ny,))
                total_white_noise += lower_bounded_tensor(param).unsqueeze(dim=1) # (ny,1)
                
        if type(total_white_noise) == float:
            return th.zeros(ny,1).double().to(device)

        return total_white_noise
        
    # def guarantee_non_neg_params(self, lower_bound:Optional[float]=1e-8) -> bool:
    #     ''' for a compounded kernel, all hyperparameters shouldn't be smaller than 0.
    #         After a update step of the kernel class, with no grad let's trim the parameters!
    #         return a bool value to notice, we have trimmed some parameters
    #     '''
    #     clamp_indicator = False
    #     with th.no_grad():
    #         for param in self.parameters():
    #             if th.any(param < lower_bound):
    #                 clamp_indicator = True
    #             param.clamp_(min=lower_bound, max=1e6)
    #     return clamp_indicator
    
    ''' the following operator class will return Compounded kernel class 
    '''
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
    
    def __repr__(self) -> str:
        ''' print the kernel information
        '''
        output = ""
        param_len = len(self._parameters)
        counter = 0
        for name, tensor in self._parameters.items():
            counter += 1
            output += name
            output += " contains parameters: "
            output += str(lower_bounded_tensor(tensor).data)
            if counter < param_len:
                output += "\n"
        return output

def initialize_param(input_value:Union[float, th.Tensor], dim_in:int, dim_out:int) -> th.Tensor:
    ''' the following kernel implementation will use this initialization function repeatedly.
        the dim_in, dim_out here is not the same as in kernel init, just denoting the dimension of the hyper
    '''
    assert dim_in!=None or dim_out!=None, "At least one dimension should be given."
    
    if type(input_value) == float:
            assert input_value > 0., "only positive parameter!"
            if dim_in == None:
                param = input_value * th.ones((dim_out,))
            elif dim_out == None:
                param = input_value * th.ones((dim_in,))
            elif dim_out != None and dim_in != None:
                param = input_value * th.ones((dim_in * dim_out))
                
    elif type(input_value) == th.Tensor:
        
        assert th.all(input_value > 0.), "only positive parameter!"
        if dim_in == None:
            param = input_value.flatten()
            assert len(param) == dim_out
        elif dim_out == None:
            param = input_value.flatten()
            assert len(param) == dim_in
        elif dim_out != None and dim_in != None:
            param = input_value.flatten()
            assert len(param) == dim_out * dim_in
    else:
        raise Exception("You must give a feasible noise. or You might have given an integer.")
    return param

class RBF(Kernel):
    ''' RBF kernel
    '''
    # counter for naming different RBF kernel
    counter = 0
    
    def __init__(self, dim_in:int, dim_out:int, l:Optional[Union[float,th.Tensor]]=5., 
                                                c:Optional[Union[float,th.Tensor]]=1.) -> None:
        ''' dim_in:     the dimension of input x
            dim_out:    the dimension of output y
            l:          the kernel length should be a float number or torch tensor
                        to initialize the kernel
            c:          sigma, the scaling factors
        '''
        super().__init__()
        l_param = initialize_param(l, dim_in=dim_in, dim_out=dim_out)
        c_param = initialize_param(c, dim_in=None, dim_out=dim_out)
            
        param_t = th.cat([c_param, l_param]).to(device).double()
        param_t = inverse_softplus(param_t)
        self.name = "rbf{}".format(RBF.counter)
        rbf_param = nn.parameter.Parameter(param_t)
        curr_parameters = Parameters(self.name, rbf_param)

        self.set_parameters(curr_parameters) # register the parameters

        self.input_dim = dim_in
        self.output_dim = dim_out
        RBF.counter += 1
    
    @staticmethod
    def rbf(param:nn.parameter.Parameter, x:th.Tensor, y:th.Tensor, diag:bool) -> th.Tensor:
        ''' x will have the shape (N, nx) for normal applications
            this class was also designed for sparse GPR, which needs to evaluate
            x - (ny, m, nx) for ny output channel, m inducing points, nx input dimension
            
            diag:       if true, only evaluate the diagonal terms. it needs x.shape == y.shape
        '''
        len_x = x.shape[0] if len(x.shape)==2 else x.shape[1] # or m
        x_dim = x.shape[1] if len(x.shape)==2 else x.shape[2]
        
        input_dim = x_dim
        output_dim = int( len(param)//(x_dim+1) )

        param_nneg = lower_bounded_tensor(param) # non-negative kernel parameters
        c = param_nneg[:output_dim]
        l = param_nneg[output_dim:]
        # sigma = theta[:output_dim].view(output_dim, 1, 1)
        l = l.view(input_dim, output_dim).unsqueeze(0) # (1, nx, ny)
        c = c.view(output_dim, 1, 1) # (ny, 1, 1)
        
        if diag:
            assert x.shape == y.shape, "they must be the same input data!"
            distance = th.zeros(output_dim, len_x).to(device).double()
            return th.exp( - distance ** 2 )
        else:
            if len(x.shape) == 3: # say (ny, m, nx)
                assert len(x.shape) == len(y.shape) == 3, "You'r using inducing variable settings, please match the dim"
                x = x.permute(1, 2, 0) # (m, nx, ny) # for each output channel there are m inducing points.
                y = y.permute(1, 2, 0) # (m, nx, ny)
            else:
                x = x.view(x.shape[0], x.shape[1], 1) # (N, nx, 1) # for normal use, for all output, they share a single data
                y = y.view(y.shape[0], y.shape[1], 1) # (N, nx, 1)
                
            x = (x/l).permute(2,0,1) # shape: (ny, N, nx)
            y = (y/l).permute(2,0,1) # shape: (ny, M, nx), let's regard ny axis as batch
            distance = th.cdist(x, y) # (ny, N, M)
            return c * th.exp( - distance ** 2 )
    
    def forward(self, x:th.Tensor, y:th.Tensor, diag:Optional[bool]=False) -> th.Tensor:
        ''' x - (n, nx), y - (h, nx) -> (ny, n, h)
            if x or y has three axis
            such as inducing variables:
            x - (ny, m, nx), y - (h, nx) -> (ny, m, h)
        '''
        theta = eval("self.{}".format(self.name))
        return RBF.rbf(theta, x, y, diag)
    

## TODO implement Periodic, RQK, linear (3 more kernels)
class White(Kernel):
    ''' white noise kernel
    '''
    counter = 0
    
    def __init__(self, dim_in:int, dim_out:int, c:Optional[Union[float,th.Tensor]]=1e-2) -> None:
        super().__init__()
        if type(c) == float:
            assert c > 0., "only positive noise!"
            param = c * th.ones((dim_out,))
        elif type(c) == th.Tensor:
            assert c.shape == (dim_out,), "check the dimension please."
            assert th.all(c > 0.), "only positive noise!"
            param = c.flatten()
        else:
            raise Exception("You must give a feasible noise. or You might have given an integer.")

        t = param.double().to(device)
        t = inverse_softplus(t)
        self.name = "white{}".format(White.counter)
        white_c = nn.parameter.Parameter(t)
        curr_parameters = Parameters(self.name, white_c)
        
        self.input_dim = dim_in
        self.output_dim = dim_out
        self.set_parameters(curr_parameters)
        White.counter += 1
        
    @staticmethod
    def white(param:nn.parameter.Parameter, x:th.Tensor, y:th.Tensor, diag:bool) -> th.Tensor:
        output_dim = len(param)
        param_nneg = lower_bounded_tensor(param)
        c = param_nneg
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
    
    def forward(self, x:th.Tensor, y:th.Tensor, diag:Optional[bool]=False) -> th.Tensor:
        ''' x - (n, nx), y - (h, nx) -> (ny, n, h)
            if x or y has three axis
            such as inducing variables:
            x - (ny, m, nx), y - (h, nx) -> (ny, m, h)
        '''
        c = eval("self.{}".format(self.name))
        return White.white(c, x, y, diag)
    
# there is no kernel class for exponent operation, therefore just leave it as a function
def exponent(param:nn.parameter.Parameter, x:th.Tensor, y:th.Tensor, diag:bool):
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
            "white":   "White.white",
            "exponent": "exponent"
    }
    
    def generate_func_dict(self) -> OrderedDict:
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
            
    def get_operation_dict(self) -> dict:
        ''' get the operation logic dict
        '''
        if "curr_parameters" not in self.__dict__.keys():
            raise KeyError("Parameters should have been set.")
        operation_dict = self.curr_parameters.operation_dict
        return operation_dict
    
    def evaluate_operation_dict(self, operation_dict:dict, x:th.Tensor, y:th.Tensor, diag:bool) -> dict:
        ''' numerical evaluation for each functional
        
            we won't evaluate white kernel in the forward, (white is not intended for product operation, otherwise you'll see a warning.)
            but instead use the kernel.noise function to facilitate the implementation of GPs
        '''
        operation_dict_copy = deepcopy(operation_dict)
        for key, operation in operation_dict_copy.items():
            operation = operation.split(" ")
            if len(operation) == 1: # which means fundamental kernels
                if "white" in operation[0]: # skip the evaluation of white kernel
                    operation_dict_copy[key] = 0.
                    # operation_dict_copy['A']
                else:
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
    def __init__(self, kernel1:Kernel, kernel2:Kernel) -> None:
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
        
    def forward(self,x:th.Tensor, y:th.Tensor, diag=False) -> th.Tensor:
        nx = x.shape[1] if len(x.shape)==2 else x.shape[2]
        assert nx == self.input_dim, "wrong dimension."
        operation_res = self.evaluate_operation_dict(self.operation_dict, x, y, diag)
        return next(reversed(operation_res.values()))

class Product(Kernel, CompoundKernel):
    ''' Product operator class
    '''  
    def check_white(kernel:Kernel) -> None:
        ''' warn the white in product operation
            as long as the white get involded in product, it's ok to do like this to discover the harm.
        '''
        for name, _ in kernel.named_parameters():
            if "white" in name:
                print("WARNING: You'r generating a kernel involves white kernel production, it will cause errors.")
                
    def __init__(self, kernel1:Kernel, kernel2:Kernel) -> None:
        super().__init__()
        # update the parameters table and form the func table
        if not isinstance(kernel1, Kernel) or not isinstance(kernel2, Kernel):
            raise TypeError("Operands should be kernels")
        
        Product.check_white(kernel1)
        Product.check_white(kernel2)
        
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
        
    def forward(self, x:th.Tensor, y:th.Tensor, diag=False) -> th.Tensor:
        nx = x.shape[1] if len(x.shape)==2 else x.shape[2]
        assert nx == self.input_dim, "wrong dimension."
        operation_res = self.evaluate_operation_dict(self.operation_dict, x, y, diag)
        return next(reversed(operation_res.values()))
    
class Exponentiation(Kernel, CompoundKernel):
    ''' Exponentiation operator class
    '''
    def check_white(kernel:Kernel) -> None:
        ''' warn the white in product operation
            as long as the white get involded in product, it's ok to do like this to discover the harm.
        '''
        for name, _ in kernel.named_parameters():
            if "white" in name:
                print("WARNING: You'r generating a kernel involves white-kernel exponentiation, it will cause errors.")
                
    def __init__(self, kernel1:Kernel, exponent_factor:float) -> None:
        super().__init__()
        # update the parameters table and form the func table
        if not isinstance(kernel1, Kernel):
            raise TypeError("Operand1 should be kernels")
        if not isinstance(exponent_factor, float):
            raise TypeError("exponent_factor should be a float")
        
        Exponentiation.check_white(kernel1)
        
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
        
    def forward(self, x:th.Tensor, y:th.Tensor, diag=False) -> th.Tensor:
        nx = x.shape[1] if len(x.shape)==2 else x.shape[2]
        assert nx == self.input_dim, "wrong dimension."
        operation_res = self.evaluate_operation_dict(self.operation_dict, x, y, diag)
        return next(reversed(operation_res.values()))