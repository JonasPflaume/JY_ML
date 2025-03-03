from abc import abstractmethod, ABC

class Classification(ABC):
    def __init__(self) -> None:
        ''' base class of regression
        '''
    
    @abstractmethod
    def fit(self):
        ''' fit the regression
        '''
    
    @abstractmethod
    def predict(self):
        ''' make prediction
        '''