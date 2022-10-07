import torch as th
import torch.nn as nn

# TODO wrap the network into the regression class, fit function should include the hyperparameters optimisation.

class MLP(nn.Module):
    ''' w1 -> w2 -> w3: three layers
        hyperparameter is designed to be optimized by bayesian optimization
    '''
    def __init__(self, hyperparam:dict):
        super().__init__()
        
        layer_num = hyperparam["layer"]
        layer_nodes = hyperparam["nodes"]
        net = []
        for i in range(layer_num-1):
            net.append(nn.Linear(layer_nodes[i], layer_nodes[i+1]))
            if i == layer_num - 2:
                pass
            else:
                net.append(nn.ReLU())
        self.net = nn.Sequential(
            *net
        )
    
    @staticmethod
    def setParams(network, decay:float) -> list:
        ''' function to set weight decay
        '''
        params_dict = dict(network.named_parameters())
        params=[]

        for key, value in params_dict.items():
            if key[-4:] == 'bias':
                params += [{'params':value, 'weight_decay':0.0}]
            else:             
                params +=  [{'params': value, 'weight_decay':decay}]
        return params
    
    def forward(self, x):
        return self.net(x)