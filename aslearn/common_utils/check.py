def has_method(obj, method_name:str):
    func = getattr(obj, method_name, None)
    assert callable(func), "This object doesn't have the desired '{}' method.".format(method_name)
    
def right_shape(X, shape:tuple):
    ''' if tuple has -1 shape, then that axis is omitted
    '''
    X_shape = X.shape
    assert len(X_shape) == len(shape), "X doesn't have enough axis."
    for i in range(len(X_shape)):
        if shape[i] == -1:
            continue
        assert X_shape[i] == shape[i], "At {0} axis, X has shape {1}, but desired {2}.".format(i, X_shape[i], shape[i])