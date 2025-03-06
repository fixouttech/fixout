import numpy as np

class FixOutArtifact:
    
    def __init__(self,
                 features_name,
                 training_data,
                 testing_data=[],
                 nonnumeric_features=[],
                 model=None, 
                 y_pred=None,
                 prob_y_pred=None, 
                 sensitive_features=[], 
                 dictionary=None):
        
        self.nonnumeric_features = nonnumeric_features
        self.features_name = features_name
        
        self.model = model
        self.y_pred = y_pred
        self.prob_y_pred = prob_y_pred
          
        self.X = training_data[0]
        self.y = training_data[1]

        if self.X is not None:
            if not isinstance(self.X, (np.ndarray)):
                self.X = np.array(self.X)
                # todo check the number of lines with X_train

        if self.y is not None:
            if not isinstance(self.y, (np.ndarray)):
                self.y = np.array(self.y)
                # todo check the number of lines with X_train
        
        self.test_data = []

        for i in range(len(testing_data)):

            X,y,label = testing_data[i]

            if X is not None:
                if not isinstance(X, (np.ndarray)):
                    X = np.array(X)

            if y is not None:
                if not isinstance(self.y, (np.ndarray)):
                    y = np.array(y)

            self.test_data.append((X,y,label))

        
        self.sensfeatList = sensitive_features
        self.dictionary = dictionary

        # if ... check if all of them are not None at the same time
        
