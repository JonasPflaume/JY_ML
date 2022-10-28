from collections import namedtuple, OrderedDict
import torch as th
import torch.nn as nn
from enum import Enum

def next_n_alpha(s, next_n):
    return chr((ord(s.upper()) + next_n - 65) % 26 + 65)

class KernelOperation(Enum):
    EXP = "exp"
    ADD = "add"
    MUL = "mul"

class Parameters(
                namedtuple('ParametersInfo', 
                ("tensor_dict", "operation_dict"))
                ):
    ''' The parameters class was design to perform the kernel operation including addition, multiplication, and exponentiation
    
        Parameters will contain a
    '''
    __slots__ = ()
    
    def __new__(cls, name, tensor, requres_grad=True):
        ''' we use number to indicate the operation sequence
        '''
        if not isinstance(name, str):
            raise TypeError("Initialize the parameters with a string of name corresponding to the kernel.")
        if not isinstance(tensor, nn.parameter.Parameter):
            raise TypeError("We use pytorch to optimize hyperparameters.")
        
        tensor.requires_grad = requres_grad
        
        tensor_dict = OrderedDict()
        tensor_dict[name] = tensor
        
        operation_dict = OrderedDict()
        operation_dict['A'] = name
        return super(Parameters, cls).__new__(cls, tensor_dict, operation_dict)
    
    def join(self, param2, operation:KernelOperation):
        ''' join two parameters instance
            parameters will be concatenated as a set.
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

class Kernel(nn.Module):
    
    def register_parameters(self, parameters_cls):
        pass
    
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
    
class Sum(Kernel):
    '''
    '''

class Product(Kernel):
    '''
    '''

class Exponentiation(Kernel):
    '''
    '''

#####################
##      Kernels    ##
#####################

class RBF(Kernel):
    '''
    '''

class Matern(Kernel):
    '''
    '''

class RQK(Kernel):
    '''
    '''

class Period(Kernel):
    '''
    '''

class Constant(Kernel):
    '''
    '''

class DotProduct(Kernel):
    '''
    '''