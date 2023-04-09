import torch as th
from aslearn.parametric.classification import Classification
device = "cuda" if th.cuda.is_available() else "cpu"

class LogisticReg(Classification):
    ''' Implementation of Logistic regression
        parameter optimized through Newton method.
        
        I didn't try to avoid potential numerical problem in newton step, 
        but they can be easily tackled by: 
            1. more data, 
            2. less features, 
            3. line search,
            4. change initialization of the weight.
    '''
    def __init__(self) -> None:
        super().__init__()
        
    def fit(self, X, Y):
        '''
            X:      (N, feature)
            Y:      (N, Label), use one-hot encodding please!
            Newton method.
            parameter space = (classNum * featureNum)
        '''
        # parameters initialization
        self.classNum = Y.shape[1]
        self.featureNum = X.shape[1]
        self.W = th.zeros(self.classNum, self.featureNum).double().to(device)
        
        stop_flag = False
        feature_outer_product = th.einsum("bi,bj->bij", X, X) # (N, featureNum, featureNum)
        
        while not stop_flag:
            
            Y_pred = self.predict(X, return_prob=True) # (N, classNum)
            # get gradient
            temp = Y_pred - Y # (N, classNum)
            gradient = th.einsum("ba,bc->bac", temp, X).sum(dim=0).view(self.classNum * self.featureNum, 1)
            
            # get hessian
            hessian = th.zeros([self.classNum * self.featureNum, self.classNum * self.featureNum]).double().to(device)
            
            for j in range(self.classNum):
                for k in range(self.classNum):
                    # This double loop cannot be easily removed by tensor operation, however, 
                    # the complexity is independent of data num, therefore, it's not expensive in most cases.
                    if k == j:
                        hessian_block = Y_pred[:,k] * (1 - Y_pred[:,j]) # (N,)
                        hessian_block = th.einsum("b,bij->ij", hessian_block, feature_outer_product)
                    else:
                        hessian_block = Y_pred[:,k] * ( - Y_pred[:,j])
                        hessian_block = th.einsum("b,bij->ij", hessian_block, feature_outer_product)
                        
                    hessian[j*self.featureNum:(j+1)*self.featureNum, k*self.featureNum:(k+1)*self.featureNum] = \
                        hessian_block.clone()
                        
            W_old = self.W.clone().view(-1,1)
            try:
                # netwon step
                W_new = W_old - th.linalg.pinv(hessian) @ gradient
                # pinv is numerically more stable. Actually we should use line search here...
            except:
                raise ValueError("Please tune the feature to make optimization well defined!")
            self.W = W_new.view(self.classNum, self.featureNum)
            
            if th.linalg.norm(gradient) < 1e-9: # give it a small tolerance
                # standard convergence criterion, check Boyd book part 3
                stop_flag = True
    
    def predict(self, x, return_prob=False):
        # get the softmax solution
        a = th.exp( self.W @ x.T ) # (classNum, N)
        pred = a / th.sum(a, dim=0)
        pred = pred.T
        if return_prob:
            return pred
        else:
            index = pred.argmax(dim=1)
            pred = th.zeros(pred.shape, dtype=int).to(device).scatter(dim=1, index=index.unsqueeze(dim=1), value=1)
            return pred
    
if __name__ == "__main__":
    from aslearn.data.get_data import classification_toy_data
    import matplotlib.pyplot as plt
    from aslearn.feature.bellcurve import BellCurve

    import numpy as np
    
    Xtrain, Ytrain, Xtest, Ytest = classification_toy_data()
    x_min, x_max = np.min(Xtrain), np.max(Xtrain)
    # visualization
    class_1_index = np.where(Ytrain[:,0]==1)
    class_2_index = np.where(Ytrain[:,1]==1)
    class_3_index = np.where(Ytrain[:,2]==1)
    plt.figure(figsize=[7,14])
    plt.subplot(211)
    plt.title("training data")
    plt.scatter(Xtrain[class_1_index,0], Xtrain[class_1_index,1], c='r', label='class 1')
    plt.scatter(Xtrain[class_2_index,0], Xtrain[class_2_index,1], c='b', label='class 2')
    plt.scatter(Xtrain[class_3_index,0], Xtrain[class_3_index,1], c='c', label='class 3')
    plt.legend()
    
    PT = BellCurve(degree=13).fit(Xtrain) # bell curve is a local feature
    Xtrain = PT(Xtrain)
    Xtest = PT(Xtest)
    
    Xtrain, Ytrain, Xtest, Ytest = th.from_numpy(Xtrain).to(device), th.from_numpy(Ytrain).to(device), \
        th.from_numpy(Xtest).to(device), th.from_numpy(Ytest).to(device)
    Lgr = LogisticReg()
    Lgr.fit(Xtrain, Ytrain)
    
    pred = Lgr.predict(Xtest)
    pred = th.all(th.eq(pred, Ytest), dim=1) 
    print("Test Acc: {:.3f}".format(pred.sum()/len(pred)))
    
    pred = Lgr.predict(Xtrain)
    pred = th.all(th.eq(pred, Ytrain), dim=1) 
    print("Train Acc: {:.3f}".format(pred.sum()/len(pred)))
    
    ### rough illustraion of decision boundary
    Xin_ = np.random.uniform(low=x_min, high=x_max, size=[5000,2])
    Xin = PT(Xin_)
    Xin = th.from_numpy(Xin).double().to(device)
    Yout = Lgr.predict(Xin)
    
    Yout = Yout.detach().cpu().numpy()
        
    # visualization
    class_1_index = np.where(Yout[:,0]==1)
    class_2_index = np.where(Yout[:,1]==1)
    class_3_index = np.where(Yout[:,2]==1)
    plt.subplot(212)
    plt.title("decision boundary")
    plt.scatter(Xin_[class_1_index,0], Xin_[class_1_index,1], c='r', label='class 1')
    plt.scatter(Xin_[class_2_index,0], Xin_[class_2_index,1], c='b', label='class 2')
    plt.scatter(Xin_[class_3_index,0], Xin_[class_3_index,1], c='c', label='class 3')
    plt.tight_layout()
    plt.show()
    