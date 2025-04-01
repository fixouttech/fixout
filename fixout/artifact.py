import numpy as np

class FixOutArtifact:
    """
    A fixout artifact.

    A class for handling training and testing data.

    See Also
    --------
    fixout.helper.FixOutHelper : A FixOut helper.

    """

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
        """
        Initializes the FixOutArtifact object with the provided data and configuration.

        Parameters:
        -----------
        features_name : list
            A list of feature names used in the dataset.
        training_data : tuple or list
            A tuple or list containing two elements:
            - A 2D array or matrix (ndarray) of training features.
            - A 1D array (ndarray) of target values.
        testing_data : list of tuples, optional
            A list of tuples containing testing data. Each tuple should be of the form (X_test, y_test, label).
            If no testing data is provided, an empty list will be used (default is None).
        nonnumeric_features : list, optional
            A list of indices of non-numeric features. Default is an empty list.
        model : object, optional
            A machine learning model to be used for predictions. Default is None.
        y_pred : array-like, optional
            Predicted target values from the model. Default is None.
        prob_y_pred : array-like, optional
            Predicted probabilities for each class from the model. Default is None.
        sensitive_features : list, optional
            A list of sensitive features used for fairness evaluation. Default is an empty list.
        dictionary : dict, optional
            A dictionary for any additional information. Default is None.

        Raises:
        -------
        ValueError
            If `training_data` is not a tuple or list containing exactly two elements (features and targets).
        """
        
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
        
