import torch as th
import torch.nn as nn
# from regression import Regression, TODO add base class regression into MLP, e.g., bayesian optimization for fit() 

class MLP(nn.Module):
    ''' w1 -> w2 -> w3: three layers
        hyperparameter is designed to be optimized by bayesian optimization
    '''
    def __init__(self, hyperparam:dict):
        ''' hyperparam contains:
            {"layer":int, "nodes":list(int), "actfunc":list(str)}
            
            in actfunc, if element is none, then there will be no actfunc
        '''
        super().__init__()
        
        layer_num = hyperparam.get("layer")
        layer_nodes = hyperparam.get("nodes")
        layer_actfunc = hyperparam.get("actfunc")
        
        assert len(layer_nodes) == layer_num == (len(layer_actfunc)+1), "use the correct shape of definition please."
        net = []
        for i in range(layer_num-1):
            net.append(nn.Linear(layer_nodes[i], layer_nodes[i+1]))
            if layer_actfunc[i] == None:
                pass
            else:
                net.append(eval("nn.{}()".format(layer_actfunc[i])))
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
    
    def save_model(self, addr:str):
        th.save(self.state_dict(), addr)
        
    def loar_model(self, addr:str):
        self.load_state_dict(th.load(addr))
        self.eval()
        
    

if __name__ == "__main__":
    hyper = {"layer":4, "nodes":[12,6,3,1], "actfunc":["ReLU", "Tanh", None]}
    net = MLP(hyperparam=hyper)
    print(net)