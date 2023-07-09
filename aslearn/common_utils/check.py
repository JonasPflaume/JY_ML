import torch as th
th.set_printoptions(precision=4, sci_mode=True)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def HAS_METHOD(obj, method_name:str):
    func = getattr(obj, method_name, None)
    assert callable(func), "This object doesn't have the desired '{}' method.".format(method_name)
    
def RIGHT_SHAPE(X, shape:tuple):
    ''' if tuple has -1 shape, then that axis is omitted
    '''
    X_shape = X.shape
    assert len(X_shape) == len(shape), "X doesn't have enough axis."
    for i in range(len(X_shape)):
        if shape[i] == -1:
            continue
        assert X_shape[i] == shape[i], "At {0} axis, X has shape {1}, but desired {2}.".format(i, X_shape[i], shape[i])
        
def WARNING(indicator:bool, info:str, info_level:int):
    ''' if indicator is true then print the info
    '''
    if indicator and info_level >= 2:
        print(f"{bcolors.WARNING}{info}{bcolors.ENDC}")
        
def REPORT_VALUE(input_value:th.Tensor, info:str, info_level:int):
    ''' if indicator is true then print the info
    '''
    if info_level >= 1:
        print("{0} {1:.4f}".format(info, input_value))
        
def PRINT(symbol:str, info_level:int):
    ''' if indicator is true then print the info
    '''
    if info_level >= 1:
        print("{0}".format(symbol), end="")