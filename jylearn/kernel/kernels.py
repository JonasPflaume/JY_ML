from abc import ABC
from aifc import Error
from ast import operator
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
                "by operation numbering: \t" + ",".join(self.operation_dict.keys()) + "\n" + \
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
        
    def set_parameters(self, parameters_cls):
        self.curr_parameters = parameters_cls
        for name, parameter in parameters_cls.tensor_dict.items():
            self.register_parameter(name, parameter)
            
    def get_parameters(self):
        if "curr_parameters" not in self.__dict__.keys():
            raise KeyError("Parameters can only be accessed after they have been set.")
        
        return self.curr_parameters
    
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

class CompoundKernel(ABC):
    ''' The base class for a compound kernel
        which contains the necessary methods to 
        reconstruct the compound kernel from individual kernels.
    '''
    def generate_func_dict(self):
        ''' generate the necessary kernel functionals
        '''
        if "curr_parameters" not in self.__dict__.keys():
            raise KeyError("Parameters should have been set.")
        tensor_dict = self.curr_parameters.tensor_dict
        func_chain_dict = OrderedDict()
        for name, param in tensor_dict.items():
            command = "".join([i for i in name if not i.isdigit()])
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
    
    def evaluate_operation_dict(self, operation_dict, x, y):
        ''' calculate the numerical evaluation for each step
        '''
        operation_dict_copy = deepcopy(operation_dict)
        for key, operation in operation_dict_copy.items():
            operation = operation.split(" ")
            if len(operation) == 1: # which means basic operation
                operation_dict_copy[key] = self.func_dict[operation[0]](x, y)
                
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
        
        self.input_dim = kernel1.input_dim
        parameters_table1 = CopiedParameter(kernel1.get_parameters())
        parameters_table2 = kernel2.get_parameters()
        
        parameters_table1.join(parameters_table2, KernelOperation.ADD)
        
        self.set_parameters(parameters_cls=parameters_table1)
        
        self.func_dict = self.generate_func_dict()
        self.operation_dict = self.get_operation_dict()
        
    def forward(self, x, y):
        assert x.shape[1] == self.input_dim, "wrong dimension."
        operation_res = self.evaluate_operation_dict(self.operation_dict, x, y)
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
        
        self.input_dim = kernel1.input_dim
        parameters_table1 = CopiedParameter(kernel1.get_parameters())
        parameters_table2 = kernel2.get_parameters()
        
        parameters_table1.join(parameters_table2, KernelOperation.MUL)
        
        self.set_parameters(parameters_cls=parameters_table1)
        
        self.func_dict = self.generate_func_dict()
        self.operation_dict = self.get_operation_dict()
        
    def forward(self, x, y):
        assert x.shape[1] == self.input_dim, "wrong dimension."
        operation_res = self.evaluate_operation_dict(self.operation_dict, x, y)
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
        parameters_table1 = CopiedParameter(kernel1.get_parameters())
        exponent_factor_t = th.tensor([exponent_factor]).to(device)
        exponent_factor_param = nn.parameter.Parameter(exponent_factor_t)
        parameters_table2 = Parameters("exponent", exponent_factor_param, requres_grad=False) # constant dosen't need grad
        
        parameters_table1.join(parameters_table2, KernelOperation.EXP)
        
        self.set_parameters(parameters_cls=parameters_table1)
        
        self.func_dict = self.generate_func_dict()
        self.operation_dict = self.get_operation_dict()
        
    def forward(self, x, y):
        assert x.shape[1] == self.input_dim, "wrong dimension."
        operation_res = self.evaluate_operation_dict(self.operation_dict, x, y)
        return next(reversed(operation_res.values()))


class RBF(Kernel):
    ''' RBF kernel
    '''
    counter = 0
    
    def __init__(self, sigma:float, l:np.ndarray, dim:int):
        super().__init__()
        param = np.concatenate([np.array([sigma]), l])
        param_t = th.from_numpy(param).to(device)
        self.rbf_name = "rbf{}".format(RBF.counter)
        rbf_param = nn.parameter.Parameter(param_t)
        curr_parameters = Parameters(self.rbf_name, rbf_param)

        self.set_parameters(curr_parameters)
        assert dim == len(l), "wrong dimension."
        self.input_dim = dim
        RBF.counter += 1
    
    def forward(self, x, y):
        assert x.shape[1] == self.input_dim, "wrong dimension."
        theta = eval("self.{}".format(self.rbf_name))
        distance = th.cdist(x/theta[1:], y/theta[1:])
        return theta[0] * th.exp( - distance ** 2 )

class Matern(Kernel):
    '''
    '''

class RQK(Kernel):
    ''' rational quadratic kernel
    '''
    counter = 0
    
    def __init__(self, sigma:float, alpha:float, l:np.ndarray, dim:int):
        super().__init__()
        param = np.concatenate([np.array([sigma]), np.array([alpha]), l])
        param_t = th.from_numpy(param).to(device)
        self.rqk_name = "rqk{}".format(RBF.counter)
        rqk_param = nn.parameter.Parameter(param_t)
        curr_parameters = Parameters(self.rqk_name, rqk_param)

        self.set_parameters(curr_parameters)
        assert dim == len(l), "wrong dimension."
        self.input_dim = dim
        RQK.counter += 1
    
    def forward(self, x, y):
        assert x.shape[1] == self.input_dim, "wrong dimension."
        theta = eval("self.{}".format(self.rqk_name))
        distance = th.cdist(x/theta[2:], y/theta[2:])
        return theta[0] * ( 1 + distance ** 2 / theta[1] ) ** (-theta[1])

class Period(Kernel):
    '''
    '''
    

class Constant(Kernel):
    ''' constant kernel
    '''
    counter = 0
    
    def __init__(self, c:float, dim):
        super().__init__()
        c = np.array([c]).reshape(-1,)
        t = th.from_numpy(c).to(device)
        self.cons_name = "cons{}".format(Constant.counter)
        cons_c = nn.parameter.Parameter(t)
        curr_parameters = Parameters(self.cons_name, cons_c)

        self.input_dim = dim
        self.set_parameters(curr_parameters)
        Constant.counter += 1
    
    def forward(self, x, y):
        assert x.shape[1] == self.input_dim, "wrong dimension."
        return eval("self.{}".format(self.cons_name))

class DotProduct(Kernel):
    ''' dot product kernel: sigma * x.T @ x
    '''
    counter = 0
    
    def __init__(self, c:float, dim:int):
        super().__init__()
        c = np.array([c]).reshape(-1,)
        t = th.from_numpy(c).to(device)
        self.dot_name = "dot{}".format(DotProduct.counter)
        dot_c = nn.parameter.Parameter(t)
        curr_parameters = Parameters(self.dot_name, dot_c)
        self.input_dim = dim
        self.set_parameters(curr_parameters)
        Constant.counter += 1
    
    def forward(self, x, y):
        assert x.shape[1] == self.input_dim, "wrong dimension."
        return eval("self.{}".format(self.dot_name)) * th.einsum("ij,kj->ik", x, y)
    
    
# Those functionals were designed to used in computational graph calling.
# This means that each kernel class will correspond to one of the functionals implemented below.

def rbf(param, x, y):
    distance = param[0] * th.cdist(x/param[1:], y/param[1:])
    return th.exp( - distance ** 2 )

def cons(param, x, y):
    return param * 1.

def dot(param, x, y):
    return param * th.einsum("ij,kj->ik", x, y)

def exponent(param, x, y):
    return param * 1.

def rqk(param, x, y):
    distance = th.cdist(x/param[2:], y/param[2:])
    return param[0] * ( 1 + distance ** 2 / param[1] ) ** (-param[1])