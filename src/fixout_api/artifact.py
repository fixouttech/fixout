import numpy as np

class FixOutArtifact:
    
    def __init__(self,
                 features_name,
                 nonnumeric_features=[],
                 model=None, 
                 X_train=None,
                 y_train=None, 
                 X_test=None, 
                 y_test=None, 
                 y_pred=None,
                 prob_y_pred=None, 
                 sensitive_features=[]):
        
        self.nonnumeric_features = nonnumeric_features
        self.features_name = features_name
        
        self.model = model
        self.y_pred = y_pred
        self.prob_y_pred = prob_y_pred

        if X_train is not None:
            if not isinstance(X_train, (np.ndarray)):
                self.X_train = np.array(X_train)
                # check the number of lines with 
            else:
                self.X_train = X_train
        else:
            self.X_train = X_train

        if y_train is not None:
            if not isinstance(y_train, (np.ndarray)):
                self.y_train = np.array(y_train)
                # check the number of lines with X_train
            else:
                self.y_train = y_train
        else:
            self.y_train = y_train
        
        if X_test is not None:
            if not isinstance(X_test, (np.ndarray)):
                self.X_test = np.array(X_test)
                # check if it is not empty
            else:
                self.X_test = X_test    
        else:
            self.X_test = None

        if y_test is not None:
            if not isinstance(y_test, (np.ndarray)):
                self.y_test = np.array(y_test)
                # check the number of lines with   
            else:
                self.y_test = y_test
        else:
            self.y_test = None
        
        
        self.sensfeatList = sensitive_features

        # if ... check if all of them are not None at the same time
        